import ast
from typing import List
from omp4py.core import clause, var_renaming, new_name, new_function_call, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="firstprivate", min_args=1, repeatable=True)
def firstprivate(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    old_local_vars = ctx.with_node.local_vars.copy()
    for arg in set(args):
        if arg not in ctx.with_node.local_vars:
            raise OmpSyntaxError(f"undeclared {arg} variable", ctx.filename, ctx.with_node)
        ctx.with_node.local_vars[arg] = "_omp_" + new_name(arg)
    var_renaming(ctx.with_node, old_local_vars)

    vars_assign = []
    # variable must exists
    for arg in set(args):
        # firstprivate variables are renamed and initialised to previous value using a shadow copy function
        arg_copy = new_function_call("_omp_runtime.var_copy")
        arg_copy.args.append(ast.Name(id=old_local_vars[arg], ctx=ast.Load()))
        vars_assign.append(ast.copy_location(ast.Assign(targets=[ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Store())],
                                      value=arg_copy), ctx.with_node))

    body[0:0] = vars_assign
    return args
