import ast
import tokenize
import inspect
import os
from io import StringIO
from typing import List, TypeVar, Set, Dict, Any
from omp4py.context import OpenMPContext, BlockContext, Directive, Clause
from omp4py.error import OmpError, OmpSyntaxError

_omp_directives: Dict[str, Directive] = {}
_omp_clauses: Dict[str, Clause] = {}
_omp_context = OpenMPContext()

_dump_source = False
_dump_ast = False
_dump_dir = os.getcwd()
_dump_prefix = "debug_"

T = TypeVar("T")


# Inner implementation of omp function when is used as decorator
def omp_parse(f: T) -> T:
    sourcecode = inspect.getsource(f)
    filename = inspect.getsourcefile(f)
    # create a fake offset to preserve the source code line number
    source_offset = "\n" * (f.__code__.co_firstlineno - 1)
    if f.__code__.co_flags & inspect.CO_NESTED:
        raise OmpError("omp decorator cannot be used with nested code")
    source_ast = ast.parse(source_offset + sourcecode, "<omp4py>")

    # global and local enviroment to compile the ast
    caller_frame = inspect.currentframe().f_back.f_back  # user -> api -> core

    transformer = OmpTransformer(source_ast, filename, caller_frame.f_globals, caller_frame.f_locals)
    # we need an import to reference OMP4py functions
    source_ast.body.insert(0, ast.ImportFrom(module="omp4py",
                                             names=[ast.alias(name="runtime", asname="_omp_runtime")],
                                             level=0))
    ast.copy_location(source_ast.body[0], source_ast.body[1])
    omp_ast = transformer.visit(source_ast)
    omp_ast = ast.fix_missing_locations(omp_ast)
    # debug before compile
    if _dump_ast:
        with open(os.path.join(_dump_dir, f"{_dump_prefix}{f.__name__}.ast"), "w") as file:
            file.write(ast.dump(omp_ast, indent=4))
    if _dump_source:
        with open(os.path.join(_dump_dir, f"{_dump_prefix}{f.__name__}.py"), "w") as file:
            file.write(ast.unparse(omp_ast))
    ompcode = compile(omp_ast, filename=filename, mode="exec")

    exec(ompcode, transformer.global_env, transformer.local_env)

    # if omp is used as function instead as decorator
    if "_omp_runtime" not in transformer.global_env:
        transformer.global_env["_omp_runtime"] = transformer.local_env["_omp_runtime"]

    return transformer.local_env[f.__name__]


# directives are declared using decorator
def directive(name: str, clauses: List[str] = None, directives: List[str] = None, min_args: int = 0, max_args: int = 0):
    clauses = [] if clauses is None else clauses
    directives = [] if directives is None else directives

    def inner(f):
        _omp_directives[name] = Directive(f, clauses, directives, min_args, max_args)
        return f

    return inner


# clauses are declared using decorator like directives
def clause(name: str, min_args: int = -1, max_args: int = -1, repeatable: bool | Any = False):
    def inner(f):
        _omp_clauses[name] = Clause(f, min_args, max_args, repeatable)
        return f

    return inner


# create a unique name using a counter
def new_name(name: str) -> str:
    return name + "_" + str(_omp_context.counter.get_and_inc(1))


# Python expression or variables used as clause arguments must be parsed
def exp_parse(src: str, ctx: BlockContext) -> ast.Expr:
    try:
        exp = ast.parse(src, "<omp4py>").body[0]

        for child in ast.walk(exp):
            if hasattr(child, "lineno"):
                delattr(child, "lineno")
            if hasattr(child, "end_lineno"):
                delattr(child, "end_lineno")

        return exp
    except Exception as ex:
        raise OmpSyntaxError(str(ex), ctx.filename, ctx.with_node)


# Check in list of variables which ones are used in a function
def filter_variables(node: ast.FunctionDef, local_vars: dict[str, str]) -> List[str]:
    # we define all local variables in an outer function and read variables of inner functions are cell vars
    outer_name = "_omp_" + node.name
    outer_module = ast.Module(body=[new_function_def(outer_name)], type_ignores=[])
    outer_f: ast.FunctionDef = outer_module.body[0]
    # create variables to be referenced in inner function
    for name in local_vars.values():
        outer_f.body.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(value=None)))
    ast.fix_missing_locations(outer_module)
    outer_f.body.append(node)

    env = dict()
    exec(compile(outer_module, '<omp4py>', mode='exec'), None, env)
    read_vars = list(env[outer_name].__code__.co_cellvars)
    # write variables are local variables inside the function
    inner_func = [f for f in env[outer_name].__code__.co_consts if hasattr(f, "co_name") and f.co_name == node.name][0]
    write_vars = inner_func.co_varnames
    # but only must be shares if variable exists before
    write_vars = [var for var in write_vars if var in local_vars]
    result_vars = list(set(read_vars + write_vars))
    # return original names instead of renamed names
    inv_mapping_vars = {v: k for k, v in local_vars.items()}
    return [inv_mapping_vars[name] for name in result_vars]


# rename variables to create private variables
def var_renaming(node: ast.With, old_local_vars: dict[str, str]):
    mapping = dict()
    for name in old_local_vars:
        if old_local_vars[name] != node.local_vars[name]:
            mapping[old_local_vars[name]] = node.local_vars[name]

    for elem in ast.walk(node):
        if isinstance(elem, ast.Name) and elem.id in mapping:
            elem.id = mapping[elem.id]


# Basic omp arg tokenizer using python tokenizer
def omp_arg_tokenizer(s: str) -> list[str]:
    raw_tokens = tokenize.generate_tokens(StringIO(s).readline)

    tokens = []
    level = 0
    pos = 0
    paren = True
    for token in raw_tokens:
        if token.string == "(":
            if level == 0:
                pos = token.end[1]
            level += 1
            paren = True
        elif token.string == ")":
            level -= 1
            if level == 0:
                tokens.append(token.line[pos:token.start[1]].strip())
                pos = token.end[1]
                tokens.append(None)
        elif level == 1 and token.string == ",":
            tokens.append(token.line[pos:token.start[1]].strip())
            pos = token.end[1]
        elif token.type == tokenize.NEWLINE:
            break
        elif level == 0:
            if not paren:
                tokens.append(None)
            tokens.append(token.string)
            paren = False
    if level > 0:
        raise ValueError()  # Message error is ignored
    if tokens[-1] is not None:
        tokens.append(None)

    return tokens


# parser to transform split tokens in directives and clauses
def omp_arg_parser(tokens: List[str]) -> List[List[str]]:
    args = list()
    result = list()
    for elem in tokens:
        if elem is None:
            result.append(args[:])
            args.clear()
        else:
            args.append(elem)
    return result


# create a function
def new_function_def(name: str) -> ast.FunctionDef:
    return ast.FunctionDef(name=name, body=[], decorator_list=[],
                           args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]))


# create a function call
def new_function_call(name: str) -> ast.Call:
    if "." in name:
        attrs = [ast.Attribute(value=None, attr=id, ctx=ast.Load()) for id in name.split(".")[::-1]]
        attrs[-1] = ast.Name(id=attrs[-1].attr, ctx=ast.Load())
        for i in range(1, len(attrs)):
            attrs[i - 1].value = attrs[i]
        func = attrs[0]
    else:
        func = ast.Name(id=name, ctx=ast.Load())

    return ast.Call(func=func, args=[], keywords=[])


# return True if the node is an omp call in the actual enviroment
def is_omp_function(node: ast.AST, global_env: dict, local_env: dict) -> bool:
    from omp4py.api import omp
    # fast lookup
    if isinstance(node, ast.Name) and node.id in global_env and global_env[node.id] == omp:
        return True

    # eval lookup
    try:
        exp = compile(ast.Expression(node), filename='<omp4py>', mode='eval')
        omp_candidate = eval(exp, global_env, local_env)
        if omp_candidate == omp:
            return True
    except:
        return False  # Ignore if function is not imported


# Searches for variables that have been declared at a node point.
class OmpVariableSearch(ast.NodeVisitor):

    def __init__(self, ctx: BlockContext):
        self.ctx: BlockContext = ctx
        self.local_vars: Dict[str, str] = dict()  # original_name -> actual_name
        self.private_vars: Set[str] = set()
        self.in_function: bool = False

    def apply(self):
        try:
            for node in self.ctx.stack[::-1]:
                if hasattr(node, "local_vars"):
                    self.in_function = True
                    self.local_vars = node.local_vars.copy()
                    self.private_vars = node.private_vars.copy()
                    self.visit(node)  # will raise StopIteration

            self.visit(self.ctx.root_node)
        except StopIteration:
            return self.local_vars, self.private_vars

    def visit_With(self, node: ast.With):
        if node == self.ctx.with_node:
            # omp variables must be ignored
            self.private_vars = {var for var in self.private_vars if not var.startswith("_omp")}

            node.local_vars = self.local_vars
            node.private_vars = self.private_vars
            # private variables are also locals
            node.local_vars.update({var: var for var in node.private_vars if var not in node.local_vars})
            raise StopIteration()

        return self.generic_visit(node)

    def visit_Attribute(self, node):
        return node

    def visit_Import(self, node: ast.Assign):
        for alias in node.names:
            self.private_vars.add(alias.name)
        return node

    def visit_ImportFrom(self, node: ast.Assign):
        return self.visit_Import(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            for elem in ast.walk(target):
                if isinstance(elem, ast.Name) and isinstance(elem.ctx, ast.Store):
                    self.private_vars.add(elem.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.in_function:
            self.private_vars.add(node.name)
        else:
            # everything is global, not local, until the first function is found
            self.in_function = True
            self.local_vars.clear()
            self.private_vars.clear()

        # Only parent function arguments are visible, otherwise can be ignored
        if any([paren == node for paren in self.ctx.stack]):
            for arg in node.args.args + [node.args.vararg] + node.args.kwonlyargs + [node.args.kwarg]:
                if arg is not None:
                    self.private_vars.add(arg.arg)
            return self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.visit_FunctionDef(node)


# Traverse python ast tree to find 'with omp(...):'
class OmpTransformer(ast.NodeTransformer):

    def __init__(self, root_node: ast.AST, filename: str, global_env: dict, local_env: dict):
        self.root_node: ast.AST = root_node
        self.filename: str = filename
        self.global_env: dict = global_env
        self.local_env: dict = local_env
        self.stack: List[ast.AST] = list()

    def visit(self, node):
        self.stack.append(node)
        new_node = super().visit(node)
        self.stack.pop()
        return new_node

    # remove @omp decorator from a class or function
    def remove_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        node.decorator_list = [exp for exp in node.decorator_list if
                               not is_omp_function(exp, self.global_env, self.local_env)]
        return self.generic_visit(node)

    # @omp decorator con be used in functions
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self.remove_decorator(node)

    # @omp decorator con be used in async functions
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.remove_decorator(node)

    # @omp decorator con be used in class
    def visit_ClassDef(self, node: ast.ClassDef):
        return self.remove_decorator(node)

    # allow directives like omp("barrier") without an empty with statement
    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and is_omp_function(node.value.func, self.global_env, self.local_env):
            omp_with = ast.With(items=[ast.withitem(context_expr=node.value)], body=[ast.Pass()])
            omp_with.fake_with = True
            return self.visit_With(omp_with)

        return self.generic_visit(node)

    # Perform OpenMP transformations if is a 'with omp(...):'
    def visit_With(self, node: ast.With):
        # Check if the with block uses omp function
        if not any([isinstance(exp.context_expr, ast.Call) and is_omp_function(exp.context_expr.func, self.global_env,
                                                                               self.local_env)
                    for exp in node.items]):
            return self.generic_visit(node)
        node.omp = True

        # With will be removed so a single omp call is allowed
        if len(node.items) > 1:
            raise OmpSyntaxError("only a omp function is allowed in the with block", self.filename, node)

        args = node.items[0].context_expr.args

        if len(args) != 1:
            raise OmpSyntaxError("only one argument is allowed in the omp function", self.filename, node)

        if not isinstance(args[0], ast.Constant):
            raise OmpSyntaxError("only a constant is allowed in the omp function", self.filename, node)

        omp_arg = args[0].value

        if not isinstance(omp_arg, str):
            raise OmpSyntaxError("only a constant string is allowed in the omp function", self.filename, node)

        if len(omp_arg.strip()) == 0:
            raise OmpSyntaxError("empty string in the omp function", self.filename, node)

        try:
            tokenized_args = omp_arg_tokenizer(omp_arg)
        except:  # Tokenizer only fails is user forget close a string o a paren
            raise OmpSyntaxError("malformed omp string", self.filename, node)

        block_ctx = BlockContext(node, self.root_node, self.filename, self.stack, self.global_env, self.local_env)

        main_directive = tokenized_args[0]
        arg_clauses = omp_arg_parser(tokenized_args[2:])
        checked_clauses = dict()

        current_directive = main_directive
        unchecked_clauses = list()
        # Check if directives and clauses exists and has the number the required number of parameters
        while current_directive is not None:
            if current_directive not in _omp_directives:
                raise OmpSyntaxError(f"'{current_directive}' directive unknown", self.filename, node)
            else:
                dir_info = _omp_directives[current_directive]
                dir_subdir = None
                for ac in arg_clauses:
                    ac_name = ac[0]
                    ac_args = ac[1:]
                    # If the clause is a known clause and is supported by the current directive
                    if ac_name in dir_info.clauses and ac_name in _omp_clauses:
                        c_info = _omp_clauses[ac_name]
                        if c_info.min_args != -1 and len(ac_args) < c_info.min_args:
                            raise OmpSyntaxError(f"{ac_name}' clause expects at least {c_info.min_args} arguments, "
                                                 f"got {len(ac_args)}", self.filename, node)
                        if c_info.max_args != -1 and len(ac_args) > c_info.max_args:
                            raise OmpSyntaxError(f"{ac_name}' clause expects at most {c_info.min_args} arguments, "
                                                 f"got {len(ac_args)}", self.filename, node)
                        if ac_name in checked_clauses:
                            if c_info.repeatable != False:  # Check if repeatable (only disable for False)
                                if c_info.repeatable != True:  # If is not a boolean is a separator
                                    checked_clauses[ac_name].append(c_info.repeatable)
                                checked_clauses[ac_name] += ac_args
                            else:
                                raise OmpSyntaxError(f"{ac_name} clause can only be used once in a directive",
                                                     self.filename, node)
                        else:
                            checked_clauses[ac_name] = ac_args

                    # Instead of a clause it can be a subdirective like for in a parallel
                    elif ac_name in dir_info.directives and ac_name in _omp_directives:
                        if ac_name in checked_clauses:
                            raise OmpSyntaxError(f"{ac_name} directive can only be used once", self.filename, node)
                        d_info = _omp_directives[ac_name]
                        if d_info.min_args != -1 and len(ac_args) < d_info.min_args:
                            raise OmpSyntaxError(f"{ac_name}' expects at least {d_info.min_args} arguments, "
                                                 f"got {len(ac_args)}", self.filename, node)
                        if d_info.max_args != -1 and len(ac_args) > d_info.max_args:
                            raise OmpSyntaxError(f"{ac_name}' expects at most {d_info.min_args} arguments, "
                                                 f"got {len(ac_args)}", self.filename, node)
                        if dir_subdir is None:
                            dir_subdir = ac_name
                        checked_clauses[ac_name] = ac_args
                    else:
                        unchecked_clauses.append(ac)
                arg_clauses = unchecked_clauses
                unchecked_clauses = list()
                current_directive = dir_subdir

        if len(arg_clauses) > 0:
            raise OmpSyntaxError(f"{arg_clauses[0][0]}' clause unknown", self.filename, node)

        node.body = _omp_directives[main_directive](node.body, checked_clauses, block_ctx)

        self.generic_visit(node)
        return node.body
