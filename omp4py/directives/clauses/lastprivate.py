import ast
from typing import List
from omp4py.core import clause, var_renaming, new_name, new_function_call, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="lastprivate", min_args=1, repeatable=True)
def lastprivate(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    old_local_vars = ctx.with_node.local_vars.copy()
    var = ast.copy_location(ctx.with_node.lastprivate, ctx.with_node)

    # variable must exists
    for arg in set(args):
        if arg not in ctx.with_node.local_vars:
            raise OmpSyntaxError(f"undeclared {arg} variable", ctx.filename, ctx.with_node)

    last_block = ast.If(test=new_function_call("_omp_runtime.lastprivate"), body=[], orelse=[])
    ast.copy_location(last_block, ctx.with_node)
    last_block.test.args.append(var)

    for arg in args:
        ctx.with_node.local_vars[arg] = "_omp_" + new_name(arg)
        last_block.body.append(ast.Assign(targets=[ast.Name(id=old_local_vars[arg], ctx=ast.Store())],
                                          value=ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Load())))
    var_renaming(ctx.with_node, old_local_vars)
    ast.copy_location(last_block, ctx.with_node)
    body.append(last_block)
    return args
