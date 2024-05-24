import ast
from typing import List
from omp4py.core import clause, BlockContext


@clause(name="untied", min_args=0, max_args=0)
def untied(node: ast.AST, args: List[str], ctx: BlockContext) -> ast.AST:
    return node
