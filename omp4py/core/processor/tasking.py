import ast
import typing

from omp4py.core.directive import names
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpDirective, OmpClause, OmpArgs, OmpItem
from omp4py.core.processor.nodes import NodeContext, check_body, check_nobody
from omp4py.core.processor.varscope import var_add, var_rename, var_update
from omp4py.core.processor.common import get_item, code_to_function, name_array


@omp_processor(names.D_TASK)
def task(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)

    task_name: str = ctx.new_id(names.D_TASK)
    task_func: ast.FunctionDef
    used_vars: list[str]
    task_func, used_vars = code_to_function(ctx, task_name, body)

    body_header: list[ast.stmt] = [task_func.body[0]]
    task_func.body = task_func.body[1:]

    data_scope: set[str] = set()
    default_scope: str = names.K_SHARED
    c_if: ast.expr = ast.Constant(value=True)

    clause: OmpClause
    item: OmpItem
    for clause in clauses:
        match str(clause):
            case names.C_IF:
                c_if = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_UNTIED:
                pass # can be silently ignored
            case names.C_SHARED:
                var_add(ctx, data_scope, clause.args.array)
            case names.C_PRIVATE:
                var_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = name_array(clause.args.array)
                body_header.extend(var_rename(ctx, task_func.body, new_vars, '__new__'))
            case names.C_FIRSTPRIVATE:
                var_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = name_array(clause.args.array)
                body_header.extend(var_rename(ctx, task_func.body, new_vars, '__copy__'))
            case names.C_DEFAULT:
                default_scope = clause.args.array[0].value


    used_vars = sorted(set(used_vars) - data_scope)
    match default_scope:
        case names.K_SHARED:
            pass
        case names.K_NONE:
            if len(used_vars) > 0:
                raise ValueError(f"'{used_vars[0]}' not specified in enclosing '{names.D_TASK}'")
        case names.K_FIRSTPRIVATE:
            others: list[str] = [var for var in used_vars]
            body_header.extend(var_rename(ctx, task_func.body, others, '__new__'))
        case names.K_PRIVATE:
            others: list[str] = [var for var in used_vars]
            body_header.extend(var_rename(ctx, task_func.body, others, '__copy__'))

    task_func.body = body_header + task_func.body

    submit: ast.Call = ctx.new_call(f'{ctx.r}.task_submit')
    submit.args.append(ast.Name(id=task_name, ctx=ast.Load()))
    submit.args.append(c_if)

    new_body: list[ast.stmt] = []
    new_body.append(task_func)
    new_body.append(ast.Expr(value=submit))

    return new_body


@omp_processor(names.D_TASK_WAIT)
def task_wait(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_nobody(ctx, body)

    task_wait: ast.Call = ctx.new_call(f'{ctx.r}.task_wait')

    return [ast.Expr(task_wait)]