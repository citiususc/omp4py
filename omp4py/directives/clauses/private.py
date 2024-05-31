import ast
from typing import List
from omp4py.core import clause, new_name, var_renaming, BlockContext


@clause(name="private", min_args=1, repeatable=True)
def private(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    old_local_vars = ctx.with_node.local_vars.copy()
    for arg in set(args):
        ctx.with_node.local_vars[arg] = "_omp_" + new_name(arg)
    var_renaming(ctx.with_node, old_local_vars)

    # private variables are renamed and initialised to None
    body.insert(0, ast.copy_location(ast.Assign(
        targets=[ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Store()) for arg in sorted(set(args))],
        value=ast.Constant(value=None)), ctx.with_node))
    return args
