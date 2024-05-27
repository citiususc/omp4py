import ast
import os
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call, new_name
from omp4py.error import OmpSyntaxError


@directive(name="atomic")
def atomic(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if len(body) != 1 or not isinstance(body[0], ast.AugAssign):
        raise OmpSyntaxError("atomic directive can only enclose a 'var binop= expr'", ctx.filename, ctx.with_node)

    binop: ast.AugAssign = body[0]
    tmp_var = new_name("_omp_var")
    # atomic is performed using a temporary var to store the expression result and a lock to the atomic assign
    tmp_assign = ast.Assign(targets=[ast.Name(id=tmp_var, ctx=ast.Store())], value=binop.value)
    atomic_assign = ast.AugAssign(target=binop.target, op=binop.op, value=ast.Name(id=tmp_var, ctx=ast.Load()))

    if isinstance(binop.target, ast.Name):
        for child in ast.walk(binop.value):
            if isinstance(child, ast.Name) and child.id == binop.target.id:
                raise OmpSyntaxError(f"atomic variable '{child.id}' cannot be on the right side", ctx.filename,
                                     ctx.with_node)

    return [ast.copy_location(tmp_assign, ctx.with_node),
            ast.copy_location(
                ast.With(items=[ast.withitem(context_expr=new_function_call("_omp_runtime.level_lock"))],
                         body=[atomic_assign]), ctx.with_node)]
