import ast
from typing import List
from omp4py.error import OmpSyntaxError
from omp4py.core import clause, BlockContext


@clause(name="default", min_args=1, max_args=1)
def default(node: None, args: List[str], ctx: BlockContext) -> bool:
    if args[0] == "shared":
        return True
    elif args[0] == "none":
        return False
    raise OmpSyntaxError("only shared and none values are supported in default clause", ctx.filename, ctx.with_node)
