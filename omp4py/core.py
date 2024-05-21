import ast
import tokenize
import inspect
from io import StringIO
from typing import List, TypeVar, Dict, Any, Tuple
from omp4py.context import OpenMPContext, BlockContext, Directive, Clause
from omp4py.error import OmpSyntaxError

_omp_directives: Dict[str, Directive] = {}
_omp_clauses: Dict[str, Clause] = {}
_omp_context = OpenMPContext()
T = TypeVar("T")


# Inner implementation of omp function when is used as decorator
def omp_parse(f: T) -> T:
    sourcecode = inspect.getsource(f)
    filename = inspect.getsourcefile(f)
    # create a fake offset to preserve the source code line number
    source_offset = "\n" * (f.__code__.co_firstlineno - 1)
    source_ast = python_parse(source_offset + sourcecode, location=True)

    # global and local enviroment to compile the ast
    caller_frame = inspect.currentframe().f_back.f_back  # user -> api -> core

    transformer = OmpTransformer(source_ast, filename, caller_frame.f_globals, caller_frame.f_locals)
    omp_ast = transformer.visit(source_ast)
    # we need an import to reference omp4py functions
    omp_ast.body.insert(0, python_parse('import omp4py as _omp_omp4py').body[0])
    ast.copy_location(omp_ast.body[0], omp_ast.body[1])
    omp_ast = ast.fix_missing_locations(omp_ast)
    ompcode = compile(omp_ast, filename=filename, mode="exec")

    exec(ompcode, transformer.global_env, transformer.local_env)

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
        return python_parse(src).body[0]
    except Exception as ex:
        raise OmpSyntaxError(str(ex)).with_ast(ctx.filename, ctx.node)


# Create ast from string with and without location
def python_parse(src: str, location=False) -> ast.Module:
    node = ast.parse(src, "<omp4py>")

    if not location:
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                delattr(child, "lineno")
            if hasattr(child, "end_lineno"):
                delattr(child, "end_lineno")

    return node

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

    # return True if the node is an omp call in the actual enviroment
    def is_omp_function(self, node: ast.AST):
        try:
            exp = compile(ast.Expression(node), filename=self.filename, mode='eval')
            omp_candidate = eval(exp, self.global_env, self.local_env)

            from omp4py.api import omp
            if omp_candidate == omp:
                return True
        except:
            return False  # Ignore if function is not imported

    # remove @omp decorator from a class or function
    def remove_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        node.decorator_list = [exp for exp in node.decorator_list if not self.is_omp_function(exp)]
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

    # Perform OpenMP transformations if is a 'with omp(...):'
    def visit_With(self, node: ast.With):
        # Check if the with block uses omp function
        if not any([isinstance(exp.context_expr, ast.Call) and self.is_omp_function(exp.context_expr.func)
                    for exp in node.items]):
            return self.generic_visit(node)

        # With will be removed so a single omp call is allowed
        if len(node.items) > 1:
            raise OmpSyntaxError("Only a omp function is allowed in the with block", self.filename, node)

        args = node.items[0].context_expr.args

        if len(args) != 1:
            raise OmpSyntaxError("Only one argument is allowed in the omp function", self.filename, node)

        if not isinstance(args[0], ast.Constant):
            raise OmpSyntaxError("Only a constant is allowed in the omp function", self.filename, node)

        omp_arg = args[0].value

        if not isinstance(omp_arg, str):
            raise OmpSyntaxError("Only a constant string is allowed in the omp function", self.filename, node)

        if len(omp_arg.strip()) == 0:
            raise OmpSyntaxError("Empty string in the omp function", self.filename, node)

        try:
            tokenized_args = omp_arg_tokenizer(omp_arg)
        except:  # Tokenizer only fails is user forget close a string o a paren
            raise OmpSyntaxError("Malformed omp string", self.filename, node)

        renamed_vars = dict()
        # load new names for renamed variables
        if hasattr(node, "renamed_vars"):
            for old_name, new_name in node.renamed_vars.items():
                # if a variable was renamed more than one, we need to update the original name
                while new_name in node.renamed_vars:
                    new_name = node.renamed_vars[new_name]
                renamed_vars[old_name] = new_name

        block_ctx = BlockContext(node, self.root_node, self.filename, self.stack,
                                 self.global_env, self.local_env, renamed_vars)

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
