import ast
from typing import List, Dict
from omp4py.core import _omp_clauses, directive, BlockContext
from omp4py.directives.parallel import create_function_block
from omp4py.error import OmpSyntaxError


@directive(name="task", clauses=["if", "untied", "default", "private", "firstprivate", "shared"])
def task(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    shared_default = True if "default" not in clauses else _omp_clauses["default"](None, clauses["default"], ctx)

    new_body = create_function_block("_omp_task","_omp_runtime.task_submit", body, clauses, ctx)

    free_vars = [var for var in ctx.with_node.used_block_local_vars if var not in ctx.with_node.used_vars]
    if not shared_default and len(free_vars) > 0:
        s = ",".join(free_vars)
        raise OmpSyntaxError(f"variables ({s}) must be declared in a data-sharing clause",
                             ctx.filename, ctx.with_node)

    return new_body

