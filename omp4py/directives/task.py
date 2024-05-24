import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext
from omp4py.directives.parallel import create_function_block


@directive(name="task")
def task(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    return create_function_block("_omp_task","_omp_runtime.task_submit", body, clauses, ctx)