import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call
from omp4py.error import OmpSyntaxError


@directive(name="ordered", max_args=-1)
def ordered(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    with_block = ast.With(items=[ast.withitem(context_expr=new_function_call("_omp_runtime.ordered"))], body=body)
    ast.copy_location(with_block, ctx.with_node)
    return [with_block]
