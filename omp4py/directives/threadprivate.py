import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext
from omp4py.error import OmpSyntaxError


@directive(name=" threadprivate", max_args=-1)
def threadprivate(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    raise OmpSyntaxError("threadprivate is not implemented", ctx.filename, ctx.with_node)
    return body