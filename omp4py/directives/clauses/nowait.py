import ast
from typing import List
from omp4py.core import clause, BlockContext


@clause(name="nowait", min_args=0, max_args=0)
def nowait(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.keywords.append(ast.keyword(arg="nowait", value=ast.Constant(value=True)))
    return node
