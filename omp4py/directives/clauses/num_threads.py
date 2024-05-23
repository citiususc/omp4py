import ast
from typing import List
from omp4py.core import clause, exp_parse, BlockContext


@clause(name="num_threads", min_args=1, max_args=1)
def num_threads(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.keywords.append(ast.keyword(arg="num_threads", value=exp_parse(args[0], ctx).value))
    return node
