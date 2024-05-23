import ast
from typing import List
from omp4py.core import clause, exp_parse,BlockContext


@clause(name="if", min_args=1, max_args=1)
def c_if(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.keywords.append(ast.keyword(arg="if_", value=exp_parse(args[0], ctx).value))
    return node
