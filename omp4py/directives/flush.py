import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext


@directive(name="flush", max_args=-1)
def flush(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    return body