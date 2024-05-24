import ast
from typing import List
from omp4py.core import clause, args_renaming, new_function_call, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="lastprivate", min_args=1, repeatable=True)
def lastprivate(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    renamed_args, new_args = args_renaming(body, args, ctx)
    var = ast.copy_location(ctx.with_node.lastprivate, ctx.with_node)

    # variable must exists
    for i, arg in enumerate(renamed_args):
        if arg not in ctx.with_node.local_vars:
            raise OmpSyntaxError(f"undeclared {arg[i]} variable", ctx.filename, ctx.with_node)

    last_block = ast.If(test=new_function_call("_omp_runtime.lastprivate"), body=[], orelse=[])
    last_block.test.args.append(var)

    for old_var, new_var in zip(renamed_args, new_args):
        last_block.body.append(ast.Assign(targets=[ast.Name(id=old_var, ctx=ast.Store())],
                                          value=ast.Name(id=new_var, ctx=ast.Load())))
    ast.copy_location(last_block, ctx.with_node)
    body.append(last_block)
    return args
