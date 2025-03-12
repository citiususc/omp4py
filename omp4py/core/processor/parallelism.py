import ast

from omp4py.core.directive import names, OmpItem, tokenizer
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpDirective, OmpClause, OmpArgs
from omp4py.core.processor.nodes import NodeContext, check_body, clause_not_implemented
from omp4py.core.processor import common


@omp_processor(names.D_PARALLEL)
def parallel(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(body)
    parallel_name: str = ctx.new_id(names.D_PARALLEL)
    parallel_func: ast.FunctionDef
    parallel_func, _ = common.code_to_function(ctx, parallel_name, body)

    body_header: list[ast.stmt] = [parallel_func.body[0]]
    parallel_func.body = parallel_func.body[1:]

    data_scope: set[str] = set()
    default_scope: str = names.K_SHARED
    c_nthreads: ast.Tuple = ast.Tuple(elts=[], ctx=ast.Load())
    c_if: ast.expr = ast.Constant(value=True)
    c_safesync: ast.expr = ast.Constant(value=-1)
    c_severity: ast.expr = ast.Constant(value=names.K_FATAL)
    c_message: ast.expr = ast.Constant(value="")
    clause: OmpClause
    item: OmpItem
    for clause in clauses:
        match str(clause):
            case names.C_SHARED:
                common.data_add(ctx, data_scope, clause.args.array)
            case names.C_PRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, parallel_func.body, new_vars, '__omp.new'))
            case names.C_FIRSTPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, parallel_func.body, new_vars, '__omp.copy'))
            case names.C_REDUCTION:
                common.data_add(ctx, data_scope, clause.args.array)
                op: OmpItem = common.get_item(clause.args.modifiers, names.M_REDUCTION_ID)
                op_name: str = op.value if op.value.isidentifier() else tokenizer.tok_name[op.tokens[0].type].lower()
                new_vars: list[str] = common.name_array(clause.args.array)
                for item in clause.args.array:
                    if isinstance(item.value, ast.Subscript):
                        raise ctx.error('Array reduction not yet supported', item.value)
                body_header.extend(common.data_rename(ctx, parallel_func.body, new_vars, f"__omp.r_{op_name}_init"))
                parallel_func.body.extend(common.data_update(ctx, clause.args.array, f"__omp.r_{op_name}_comb"))
            case names.C_DEFAULT:
                default_scope = clause.args.array[0].value
            case names.C_NUM_THREADS:
                for item in clause.args.array:
                    c_nthreads.elts.append(ctx.cast_expression("int", item.value))
            case names.C_IF:
                c_if = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_SAFESYNC:
                c_safesync = clause.args.array[0].value
            case names.C_SEVERITY:
                c_severity = clause.args.array[0].value
            case names.C_MESSAGE:
                c_message = clause.args.array[0].value
            case names.C_ALLOCATE:
                raise clause_not_implemented(clause)
            case names.C_PROC_BIND:
                raise clause_not_implemented(clause)
            case names.C_COPYIN:
                raise clause_not_implemented(clause)

    match default_scope:
        case names.K_SHARED:
            pass
        case names.K_NONE:
            var: str
            body_header[0].names = [var for var in body_header[0].names if var in data_scope]
        case names.K_FIRSTPRIVATE:
            others: list[str] = [var for var in body_header[0].names if var not in data_scope]
            body_header.extend(common.data_rename(ctx, parallel_func.body, others, '__omp.new'))
        case names.K_PRIVATE:
            others: list[str] = [var for var in body_header[0].names if var not in data_scope]
            body_header.extend(common.data_rename(ctx, parallel_func.body, others, '__omp.copy'))

    parallel_func.body = body_header + parallel_func.body

    parallel_call: ast.Call = ctx.new_call("__omp.parallel_run")
    parallel_call.args.append(ast.Name(id=parallel_name, ctx=ast.Load()))
    parallel_call.args.append(c_if)
    parallel_call.args.append(c_message)
    parallel_call.args.append(c_nthreads)
    # parallel_call.args.append(c_proc_bind)
    parallel_call.args.append(c_safesync)
    parallel_call.args.append(c_severity)

    new_body: list[ast.stmt] = []
    new_body.append(parallel_func)
    new_body.append(ast.Expr(value=parallel_call))

    return new_body


def teams_callback(ctx: NodeContext, child: OmpDirective) -> None:
    if child.name not in (names.D_PARALLEL, names.D_DISTRIBUTE):
        raise ctx.error("only 'distribute' or 'parallel' regions are allowed to be strictly nested inside "
                        "'teams' region", ctx.directive)


@omp_processor(names.D_TEAMS)
def teams(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    ctx.directive_callback = teams_callback
    check_body(body)
    teams_name: str = ctx.new_id(names.D_TEAMS)
    teams_func: ast.FunctionDef
    teams_func, _ = common.code_to_function(ctx, teams_name, body)

    body_header: list[ast.stmt] = [teams_func.body[0]]
    teams_func.body = teams_func.body[1:]

    data_scope: set[str] = set()
    default_scope: str = names.K_SHARED
    c_if: ast.expr = ast.Constant(value=True)
    c_nteams: ast.Tuple = ast.Tuple(elts=[ast.Constant(value=1), ast.Constant(value=1)], ctx=ast.Load())
    c_thlimit: ast.expr = ast.Constant(value=-1)
    clause: OmpClause
    for clause in clauses:
        match str(clause):
            case names.C_SHARED:
                common.data_add(ctx, data_scope, clause.args.array)
            case names.C_PRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, teams_func.body, new_vars, '__omp.new'))
            case names.C_FIRSTPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, teams_func.body, new_vars, '__omp.copy'))
            case names.C_REDUCTION:
                common.data_add(ctx, data_scope, clause.args.array)
                op: OmpItem = common.get_item(clause.args.modifiers, names.M_REDUCTION_ID)
                op_name: str = op.value if op.value.isidentifier() else tokenizer.tok_name[op.tokens[0].type].lower()
                new_vars: list[str] = common.name_array(clause.args.array)
                item: OmpItem
                for item in clause.args.array:
                    if isinstance(item.value, ast.Subscript):
                        raise ctx.error('Array reduction not yet supported', item.value)
                body_header.extend(common.data_rename(ctx, teams_func.body, new_vars, f"__omp.r_{op_name}_init"))
                teams_func.body.extend(common.data_update(ctx, clause.args.array, f"__omp.r_{op_name}_comb"))
            case names.C_DEFAULT:
                default_scope = clause.args.array[0].value
            case names.C_IF:
                c_if = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_THREAD_LIMIT:
                c_thlimit = ctx.cast_expression("int", clause.args.array[0].value)
            case names.C_NUM_TEAMS:
                c_nteams.elts[1] = ctx.cast_expression("int", clause.args.array[0].value)
                lower_bound: OmpItem = common.get_item(clause.args.modifiers, names.M_LOWER_BOUND)
                if lower_bound is not None:
                    c_nteams.elts[0] = ctx.cast_expression("int", lower_bound.value)
            case names.C_ALLOCATE:
                raise clause_not_implemented(clause)

    match default_scope:
        case names.K_SHARED:
            pass
        case names.K_NONE:
            var: str
            body_header[0].names = [var for var in body_header[0].names if var in data_scope]
        case names.K_FIRSTPRIVATE:
            others: list[str] = [var for var in body_header[0].names if var not in data_scope]
            body_header.extend(common.data_rename(ctx, teams_func.body, others, '__omp.new'))
        case names.K_PRIVATE:
            others: list[str] = [var for var in body_header[0].names if var not in data_scope]
            body_header.extend(common.data_rename(ctx, teams_func.body, others, '__omp.copy'))

    teams_func.body = body_header + teams_func.body

    teams_call: ast.Call = ctx.new_call("__omp.teams_run")
    teams_call.args.append(ast.Name(id=teams_name, ctx=ast.Load()))
    teams_call.args.append(c_if)
    teams_call.args.append(c_nteams)
    teams_call.args.append(c_thlimit)

    new_body: list[ast.stmt] = []
    new_body.append(teams_func)
    new_body.append(ast.Expr(value=teams_call))

    return new_body
