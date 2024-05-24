import ast
from typing import List
from omp4py.core import clause, BlockContext


@clause(name="ordered", min_args=0, max_args=0,repeatable=True)
def ordered(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.keywords.append(ast.keyword(arg="ordered", value=ast.Constant(value=True)))
    return node