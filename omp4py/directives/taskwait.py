import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call
from omp4py.error import OmpSyntaxError


@directive(name="taskwait")
def taskwait(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if not (len(body) == 1 and isinstance(body[0], ast.Pass)):
        raise OmpSyntaxError("taskwait directive only allows pass as body", ctx.filename, ctx.with_node)
    return body + [ast.copy_location(ast.Expr(value=new_function_call("_omp_runtime.taskwait")), ctx.with_node)]
