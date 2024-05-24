import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call


@directive(name="critical", max_args=-1)
def critical(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    return [ast.copy_location(
        ast.With(items=[ast.withitem(context_expr=new_function_call("_omp_runtime.level_lock"))],
                 body=body), ctx.with_node)]
