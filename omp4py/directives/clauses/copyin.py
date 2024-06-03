import ast
from typing import List
from omp4py.core import clause, BlockContext, new_function_call, new_name


@clause(name="copyin", min_args=1)
def copyin(body: List[ast.AST], args: List[str], ctx: BlockContext):
    # use a runtime function to share threadprivate from master
    copyin = new_function_call("_omp_runtime.copyin")
    copyin.args = [ast.Constant(value=new_name("_omp_copyin"))] + [ast.Constant(value=arg) for arg in args]
    body.insert(0, ast.copy_location(ast.Expr(value=copyin), ctx.with_node))
