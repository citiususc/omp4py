import ast
from typing import List
from omp4py.core import clause, exp_parse, new_function_call,BlockContext


@clause(name="schedule", min_args=1, max_args=2)
def schedule(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.func = new_function_call("_omp_runtime.omp_range").func

    node.keywords.append(ast.keyword(arg="schedule", value=ast.Constant(value=args[0])))
    if len(args) > 1:
        node.keywords.append(ast.keyword(arg="chunk_size", value=exp_parse(args[1], ctx).value))

    return node
