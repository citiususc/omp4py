import ast
from typing import List, Dict
from omp4py.context import BlockContext
from omp4py.error import OmpSyntaxError
from omp4py.core import directive, _omp_clauses, _omp_directives, filter_variables, OmpVariableSearch, \
    new_name, new_function_call, new_function_def


def create_function_block(name: str, runner: str, body: List[ast.AST], clauses: Dict[str, List[str]],
                          ctx: BlockContext) -> List[ast.AST]:
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
    ctx.with_node.local_vars = OmpVariableSearch(ctx).local_vars
    block_local_vars = filter_variables(omp_parallel_block, ctx.with_node.local_vars)
    shared_default = True if "default" not in clauses else _omp_clauses["default"](None, clauses["default"], ctx)

    # clauses that affect to variables
    for clause in ["shared", "private", "firstprivate", "reduction"]:
        if clause in clauses:
            for var in clauses[clause]:
                if var in used_vars:
                    raise OmpSyntaxError(f"Variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                           "simultaneously", ctx.filename, ctx.with_node)
            vars_in_clause = _omp_clauses[clause](body, clauses[clause], ctx)
            used_vars.update({v: clause for v in vars_in_clause})

    # we declare remainder variables as shared or raise an error if default is none
    free_vars = [var for var in block_local_vars if var not in used_vars]
    if shared_default:
        if len(free_vars) > 0:
            _omp_clauses["shared"](body, free_vars, ctx)
    elif len(free_vars) > 0:
        s = ",".join(free_vars)
        raise OmpSyntaxError(f"Variables ({s}) must be declared shared, private or firstprivate clauses",
                               ctx.filename, ctx.with_node)

    if "if" in clauses:
        _omp_clauses["if"](omp_parallel_call, clauses["if"], ctx)

    if "num_threads" in clauses:
        _omp_clauses["num_threads"](omp_parallel_call, clauses["num_threads"], ctx)

    return new_body


@directive(name="parallel", clauses=["if", "num_threads", "default", "private", "firstprivate", "shared", "reduction"],
           directives=["for", "sections"])
def parallel(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    body_start = body[0]
    body_end = body[-1]
    new_body = create_function_block("_omp_parallel", "_omp_runtime.parallel_run", body, clauses, ctx)

    if "for" not in clauses and "sections" not in clauses:
        return new_body

    # parallel can be combined with for or sections, used clauses had been removed
    subdir_args = {c: a for c, a in clauses.items() if c not in _omp_directives["parallel"].clauses}
    # hide changes to sub directive
    body_start_i = body.index(body_start)
    body_end_i = body.index(body_end) + 1
    if "for" in clauses:
        del subdir_args["for"]
        body[body_start_i:body_end_i] = _omp_directives["for"](body[body_start_i:body_end_i], subdir_args, ctx)
    elif "sections" in clauses:
        del subdir_args["sections"]
        body[body_start_i:body_end_i] = _omp_directives["sections"](body[body_start_i:body_end_i], subdir_args, ctx)

    return new_body
