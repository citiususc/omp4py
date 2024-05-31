import ast
from typing import List
from omp4py.core import clause, exp_parse, new_function_call, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="schedule", min_args=1, max_args=2)
def schedule(node: ast.Call, args: List[str], ctx: BlockContext) -> ast.Call:
    node.func = new_function_call("_omp_runtime.omp_range").func

    if args[0] not in ['static', 'dynamic', 'guided', 'runtime', 'auto']:
        raise OmpSyntaxError("schedule must be one of 'static', 'dynamic', 'guided', 'runtime', 'auto'",
                             ctx.filename, ctx.with_node)

    if args[0] in ['runtime', 'auto'] and len(args) == 2:
        raise OmpSyntaxError("when schedule(runtime) or schedule(auto) is specified, "
                             "chunk_size must not be specified", ctx.filename, ctx.with_node)

    node.keywords.append(ast.keyword(arg="schedule", value=ast.Constant(value=args[0])))
    if len(args) > 1:
        node.keywords.append(ast.keyword(arg="chunk_size", value=exp_parse(args[1], ctx).value))

    return node
