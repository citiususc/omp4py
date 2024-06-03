import ast
from typing import List, Dict
from omp4py.context import BlockContext
from omp4py.error import OmpSyntaxError
from omp4py.core import directive, _omp_clauses, _omp_directives, filter_variables, OmpVariableSearch, \
    new_name, new_function_call, new_function_def


# search for return/yield inside omp parallel to raise error
class OmpReturnSearch(ast.NodeVisitor):

    def __init__(self, ctx: BlockContext):
        self.ctx: BlockContext = ctx
        self.visit(ctx.with_node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Return(self, node: ast.Return):
        raise OmpSyntaxError("parallel directive block cannot contain return statements", self.ctx.filename, node)

    def visit_Yield(self, node: ast.Yield):
        raise OmpSyntaxError("parallel directive block cannot contain yield statements", self.ctx.filename, node)

    def visit_YieldFrom(self, node: ast.YieldFrom):
        raise OmpSyntaxError("parallel directive block cannot contain yield from statements", self.ctx.filename, node)


def create_function_block(name: str, runner: str, body: List[ast.AST], clauses: Dict[str, List[str]],
                          ctx: BlockContext) -> List[ast.AST]:
    OmpReturnSearch(ctx)

    new_body = list()
    function_name = new_name(name)

    # to execute in parallel, the code must be enclosed in a function
    omp_parallel_block = new_function_def(function_name)
    ast.copy_location(omp_parallel_block, ctx.with_node)
    ast.fix_missing_locations(omp_parallel_block)
    omp_parallel_block.body = body
    new_body.append(omp_parallel_block)

    # create a call to execute the block in parallel
    omp_parallel_call = new_function_call(runner)
    omp_parallel_call.args.append(ast.Name(id=function_name, ctx=ast.Load()))
    ast.copy_location(omp_parallel_call, ctx.with_node)
    new_body.append(ast.Expr(value=omp_parallel_call))

    # we need to handle variables
    used_vars = dict()
    OmpVariableSearch(ctx).apply()
    used_block_local_vars = filter_variables(omp_parallel_block, ctx.with_node.local_vars)

    # assume that all private variables are now shared in new parallel block
    if hasattr(ctx.with_node, "new_team"):
        ctx.with_node.private_vars.clear()

    # clauses that affect to variables
    for clause in ["shared", "private", "firstprivate", "reduction"]:
        if clause in clauses:
            vars_in_clause = _omp_clauses[clause](body, clauses[clause], ctx)
            for var in vars_in_clause:
                if var in used_vars:
                    raise OmpSyntaxError(f"variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                         "simultaneously", ctx.filename, ctx.with_node)
            if clause != "shared":
                ctx.with_node.private_vars.update(vars_in_clause)
            used_vars.update({v: clause for v in vars_in_clause})

    if "copyin" in clauses:
        _omp_clauses["copyin"](body, clauses["copyin"], ctx)

    # subdirective need used vars
    ctx.with_node.used_vars = used_vars.copy()

    # we declare remainder variables as shared or raise an error if default is none
    free_vars = [var for var in used_block_local_vars if var not in used_vars]
    ctx.with_node.used_block_local_vars = used_block_local_vars
    if len(free_vars) > 0:
        _omp_clauses["shared"](body, free_vars, ctx)

    if "if" in clauses:
        _omp_clauses["if"](omp_parallel_call, clauses["if"], ctx)

    if "num_threads" in clauses:
        _omp_clauses["num_threads"](omp_parallel_call, clauses["num_threads"], ctx)

    return new_body


@directive(name="parallel",
           clauses=["if", "num_threads", "default", "private", "firstprivate", "shared", "reduction", "copyin"],
           directives=["for", "sections"])
def parallel(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    shared_default = True if "default" not in clauses else _omp_clauses["default"](None, clauses["default"], ctx)
    ctx.with_node.new_team = True
    body_start = body[0]
    body_end = body[-1]

    # parallel can be combined with for or sections
    parallel_clauses = {c: a for c, a in clauses.items() if c in _omp_directives["parallel"].clauses}
    subdir_args = {c: a for c, a in clauses.items() if c not in _omp_directives["parallel"].clauses}

    # with sections reduction must be done in each section
    if "sections" in clauses and "reduction" in clauses:
        subdir_args["reduction"] = clauses["reduction"]
        del parallel_clauses["reduction"]

    new_body = create_function_block("_omp_parallel", "_omp_runtime.parallel_run", body, parallel_clauses, ctx)

    # hide changes to sub directive
    body_start_i = body.index(body_start)
    body_end_i = body.index(body_end) + 1
    if "for" in clauses:
        del subdir_args["for"]
        body[body_start_i:body_end_i] = _omp_directives["for"](body[body_start_i:body_end_i], subdir_args, ctx)
    elif "sections" in clauses:
        del subdir_args["sections"]
        body[body_start_i:body_end_i] = _omp_directives["sections"](body[body_start_i:body_end_i], subdir_args, ctx)

    free_vars = [var for var in ctx.with_node.used_block_local_vars if var not in ctx.with_node.used_vars]
    if not shared_default and len(free_vars) > 0:
        s = ",".join(free_vars)
        raise OmpSyntaxError(f"variables ({s}) must be declared in a data-sharing clause",
                             ctx.filename, ctx.with_node)

    return new_body
