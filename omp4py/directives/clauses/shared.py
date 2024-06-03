import ast
from typing import List
from omp4py.core import clause, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="shared", min_args=1, repeatable=True)
def shared(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    # variable must exists
    for arg in args:
        if arg not in ctx.with_node.local_vars:
            raise OmpSyntaxError(f"undeclared {arg} variable", ctx.filename, ctx.with_node)

    body.insert(0, ast.Nonlocal(names=sorted(set(args))))
    return args
