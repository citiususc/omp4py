import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call
from omp4py.error import OmpSyntaxError


@directive(name="barrier")
def barrier(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if not (len(body) == 1 and isinstance(body[0], ast.Pass)):
        raise OmpSyntaxError("barrier directive only allows pass as body", ctx.filename, ctx.with_node)
    return [ast.copy_location(ast.Expr(value=new_function_call("_omp_runtime.barrier")), ctx.with_node)]
