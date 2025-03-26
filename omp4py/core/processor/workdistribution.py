import ast
import typing

from omp4py.core.directive import names, tokenizer
from omp4py.core.processor.common import get_item
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpClause, OmpArgs, OmpItem
from omp4py.core.processor import common
from omp4py.core.processor.nodes import (NodeContext, Variables, node_name, directive_node, check_body,
                                         clause_not_implemented)

__all__ = []

@omp_processor(names.D_SINGLE)
def single(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)
    old_variables: Variables = ctx.variables.new_scope()
    single_call: ast.Call = ctx.new_call(f'{ctx.r}.single')
    single_call.args.append(ast.Name(id='__name__', ctx=ast.Load()))
    single_call.args.append(ast.Constant(value=ctx.directive.lineno))
    single_call.args.append(ast.Constant(value=ctx.directive.col_offset))

    body_header: list[ast.stmt] = list()

    data_scope: set[str] = set()
    c_pcopy: list[str] = list()
    c_nowait: ast.expr = ast.Constant(value=False)
    clause: OmpClause
    item: OmpItem
    for clause in clauses:
        match str(clause):
            case names.C_COPYPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                for item in clause.args.array:
                    c_pcopy.append(item.value)
            case names.C_PRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.new'))
            case names.C_FIRSTPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.copy'))
            case names.C_NOWAIT:
                if clause.args is None:
                    c_nowait = ast.Constant(value=True)
                else:
                    c_nowait = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_ALLOCATE:
                pass  # TODO

    single_if: ast.If = ctx.copy_pos(ast.If(test=single_call, body=body_header + body, orelse=[]))

    if len(c_pcopy) > 0:
        copyprivate_name: str = ctx.new_id("copyprivate")
        cp_write: ast.Call = ctx.new_call(f'{ctx.r}copyprivate_write')
        cp_read: ast.Call = ctx.new_call(f'{ctx.r}copyprivate_read')
        cp_write.args.append(c_nowait)
        cp_read.args.append(c_nowait)

        cp_code: ast.FunctionDef = ctx.new_function(copyprivate_name)
        refs: ast.Nonlocal = ast.Nonlocal(names=list())
        cp_code.body.append(refs)
        cp_read.args.append(ast.Name(id=copyprivate_name, ctx=ast.Load()))

        name: str
        for name in c_pcopy:
            var: str = ctx.variables.final_name(name)
            arg_var: str = f"__omp_{var}"
            cp_write.args.append(ast.Name(id=var, ctx=ast.Load()))
            cp_code.args.args.append(ast.arg(arg_var))
            refs.names.append(var)
            cp_code.body.append(ast.Assign(targets=[ast.Name(id=var, ctx=ast.Store())],
                                           value=ast.Name(id=arg_var, ctx=ast.Load())))
        single_if.body.append(ast.Expr(cp_write))
        single_if.orelse.append(cp_code)
        single_if.orelse.append(ast.Expr(cp_read))
    else:
        single_if.body.append(common.no_wait(ctx, c_nowait))
        single_if.orelse.append(common.no_wait(ctx, c_nowait))

    single_if.body.extend(common.data_delete(ctx, old_variables))

    ctx.variables = old_variables
    return [single_if]


def is_section(ctx: NodeContext, stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.With) and \
        isinstance(stmt.items[0].context_expr, ast.Call) and \
        ctx.is_omp(stmt.items[0].context_expr) and \
        directive_node(ctx, stmt.items[0].context_expr).name == names.D_SECTION


@omp_processor(names.D_SECTIONS)
def sections(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)
    old_variables: Variables = ctx.variables.new_scope()
    new_body: list[ast.stmt] = list()
    body_header: list[ast.stmt] = list()
    body_footer: list[ast.stmt] = list()

    data_scope: set[str] = set()
    c_nowait: ast.expr = ast.Constant(value=False)
    clause: OmpClause
    item: OmpItem
    for clause in clauses:
        match str(clause):
            case names.C_PRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.new'))
            case names.C_FIRSTPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.copy'))
            case names.C_REDUCTION:
                common.data_add(ctx, data_scope, clause.args.array)
                op: OmpItem = common.get_item(clause.args.modifiers, names.M_REDUCTION_ID)
                op_name: str = op.value if op.value.isidentifier() else tokenizer.tok_name[op.tokens[0].type].lower()
                new_vars: list[str] = common.name_array(clause.args.array)
                for item in clause.args.array:
                    if isinstance(item.value, ast.Subscript):
                        raise ctx.error('Array reduction not yet supported', item.value)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.r_{op_name}_init'))
                body_footer.extend(common.data_update(ctx, clause.args.array, f'{ctx.r}.r_{op_name}_comb'))
            case names.C_NOWAIT:
                if clause.args is None:
                    c_nowait = ast.Constant(value=True)
                else:
                    c_nowait = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_LASTPRIVATE:
                raise clause_not_implemented(clause)
            case names.C_ALLOCATE:
                raise clause_not_implemented(clause)

    stmt: ast.stmt
    i: int
    for i, stmt in enumerate(body):
        sections_call: ast.Call = ctx.new_call(f'{ctx.r}.section')
        sections_call.args.append(ast.Constant(value=i))
        sections_call.args.append(ast.Constant(value=len(body) - 1))

        if is_section(ctx, stmt):
            if len(stmt.items) > 1:
                stmt.items = stmt.items[1:]
                new_body.append(ctx.copy_pos(ast.If(test=sections_call, body=[stmt], orelse=[])))
            else:
                new_body.append(ctx.copy_pos(ast.If(test=sections_call, body=stmt.body, orelse=[])))
        else:
            raise ctx.error("expected 'omp section'", stmt)

    body_footer.extend(common.data_delete(ctx, old_variables))
    body_footer.append(common.no_wait(ctx, c_nowait))

    ctx.variables = old_variables
    return body_header + new_body + body_footer


@omp_processor(names.D_SECTION)
def section(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    raise ValueError("'omp section' may only be used in 'omp sections'")


@omp_processor(names.D_FOR)
def for_(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_body(ctx, body)

    if not isinstance(body[0], ast.For):
        raise ctx.error("for statement expected", body[0])

    if len(body) > 1:
        raise ctx.error("unindent expected, but statement found", body[1])

    old_variables: Variables = ctx.variables.new_scope()
    body_header: list[ast.stmt] = []
    body_footer: list[ast.stmt] = []
    new_body: list[ast.stmt] = []

    for_init: ast.Call = ctx.new_call(f'{ctx.r}.for_init')
    for_bounds: ast.Call = ctx.new_call(f'{ctx.r}.for_bounds')
    name_bound: str = ctx.new_id("bounds")

    new_body.append(ctx.copy_pos(ast.Assign(targets=[ast.Name(id=name_bound, ctx=ast.Store())], value=for_bounds)))
    new_body.append(ctx.copy_pos(ast.Expr(for_init)))

    for_chunk: ast.Call = ctx.new_call(f'{ctx.r}.for_next')
    while_chunk: ast.While = ast.While(test=for_chunk, body=[], orelse=[])
    while_chunk.body = body
    for_chunk.args.append(ast.Name(id=name_bound, ctx=ast.Load()))
    new_body.append(ctx.copy_pos(while_chunk))

    data_scope: set[str] = set()
    c_collapse: int = 1
    c_nowait: ast.expr = ast.Constant(value=False)
    c_order: ast.Constant = ast.Constant(value=-1)
    c_ordered: ast.expr = ast.Constant(value=0)
    c_sch_kind: ast.expr = ast.Constant(value=-1)
    c_sch_chunk: ast.expr = ast.Constant(value=-1)
    c_sch_monotonic: ast.expr = ast.Constant(value=True)
    clause: OmpClause
    item: OmpItem
    for clause in clauses:

        match str(clause):
            case names.C_COLLAPSE:
                c_collapse = int(clause.args.array[0].value)
            case names.C_SCHEDULE:
                options: list[str] = [names.K_STATIC, names.K_DYNAMIC, names.K_GUIDED, names.K_AUTO, names.K_RUNTIME]
                schedule: str = clause.args.array[0].value
                c_sch_kind = ast.Constant(value=options.index(schedule))
                if len(clause.args.array) == 2:
                    if schedule in [names.K_AUTO, names.K_RUNTIME]:
                        raise clause.args.array[0].tokens[0].\
                            make_error(f"schedule '{schedule}' does not take a 'chunk_size' parameter")
                    c_sch_chunk = ctx.cast_expression('int', clause.args.array[1].value)
                if get_item(clause.args.modifiers, names.K_NONMONOTONIC):
                    ast.expr = ast.Constant(value=False)

            case names.C_ORDER:
                c_order = ast.Constant(value=0 if clause.args.array[0].value == names.K_REPRODUCIBLE else 1)
            case names.C_ORDERED:
                if clause.args is None:
                    c_ordered = ast.Constant(value=1)
                else:
                    c_ordered = ctx.cast_expression('int', clause.args.array[0].value)
            case names.C_NOWAIT:
                if clause.args is None:
                    c_nowait = ast.Constant(value=True)
                else:
                    c_nowait = ctx.cast_expression("bool", clause.args.array[0].value)
            case names.C_PRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.new'))
            case names.C_FIRSTPRIVATE:
                common.data_add(ctx, data_scope, clause.args.array)
                new_vars: list[str] = common.name_array(clause.args.array)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.copy'))
            case names.C_LASTPRIVATE:
                raise clause_not_implemented(clause)
            case names.C_REDUCTION:
                common.data_add(ctx, data_scope, clause.args.array)
                op: OmpItem = common.get_item(clause.args.modifiers, names.M_REDUCTION_ID)
                op_name: str = op.value if op.value.isidentifier() else tokenizer.tok_name[op.tokens[0].type].lower()
                new_vars: list[str] = common.name_array(clause.args.array)
                for item in clause.args.array:
                    if isinstance(item.value, ast.Subscript):
                        raise ctx.error('Array reduction not yet supported', item.value)
                body_header.extend(common.data_rename(ctx, body, new_vars, f'{ctx.r}.r_{op_name}_init'))
                body_footer.extend(common.data_update(ctx, clause.args.array, f'{ctx.r}.r_{op_name}_comb'))
            case names.C_INDUCTION:
                raise clause_not_implemented(clause)
            case names.C_LINEAR:
                raise clause_not_implemented(clause)
            case names.C_ALLOCATE:
                raise clause_not_implemented(clause)

    bounds_list: ast.List = ast.List(elts=[], ctx=ast.Load())
    for_bounds.args.append(bounds_list)

    i: int
    c_loop: ast.stmt = body[0]
    for i in range(1, c_collapse):
        if not isinstance(c_loop.body[0], ast.For):
            raise ctx.error("for statement expected", c_loop.body[0])

        if len(c_loop.body) > 1:
            raise ctx.error("the loops must be perfectly nested", c_loop.body[1])

        c_loop = c_loop.body[0]

    for i in range(c_collapse):
        loop: ast.For = typing.cast(ast.For, body[0])

        if not isinstance(loop.iter, ast.Call) or node_name(loop.iter) != 'range':
            raise ctx.error("range for expected", loop.iter)

        if len(loop.iter.args) == 1:
            bounds_list.elts.append(ast.Constant(value=0))
        bounds_list.elts.extend(loop.iter.args)
        constant_step: bool = True
        if len(loop.iter.args) != 3:
            bounds_list.elts.append(ast.Constant(value=1))
        else:
            constant_step = common.is_constant(loop.iter.args[2])
        loop.iter.args.clear()

        if c_collapse > 1:
            loop.iter.args.append(ast.BinOp(left=ctx.array_pos(name_bound, 2 + 6 * i, ast.Load()), op=ast.Add(),
                                            right=ast.IfExp(
                                                test=ast.Compare(left=ctx.array_pos(name_bound, 0, ast.Load()),
                                                                 ops=[ast.Eq()],
                                                                 comparators=[
                                                                     ctx.array_pos(name_bound, 1, ast.Load())]),
                                                body=ctx.array_pos(name_bound, 5 + 6 * i, ast.Load()),
                                                orelse=ast.Constant(value=0))))
            loop.iter.args.append(ctx.array_pos(name_bound, 3 + 6 * i, ast.Load()))

            if i == c_collapse - 1:
                loop.body.append(ast.AugAssign(target=ctx.array_pos(name_bound, 0, ast.Store()),
                                               op=ast.Sub(), value=ast.Constant(value=1)))
            loop.body.append(ast.If(test=ast.UnaryOp(op=ast.Not(),
                                                     operand=ctx.array_pos(name_bound, 0, ast.Load())),
                                    body=[ast.Break()], orelse=[]))
        else:
            loop.iter.args.append(ctx.array_pos(name_bound, 6 * i, ast.Load()))
            loop.iter.args.append(ctx.array_pos(name_bound, 6 * i + 1, ast.Load()))

        loop.iter.args.append(ctx.clone(bounds_list.elts[-1]))
        body = loop.body

    for_init.args.append(ast.Name(id=name_bound, ctx=ast.Load()))
    for_init.args.append(c_sch_kind)
    for_init.args.append(c_sch_chunk)
    for_init.args.append(c_sch_monotonic)
    for_init.args.append(c_ordered)
    for_init.args.append(c_order)

    body_footer.extend(common.data_delete(ctx, old_variables))
    body_footer.append(common.no_wait(ctx, c_nowait))

    ctx.variables = old_variables
    return body_header + new_body + body_footer
