import ast

from omp4py.core.parser.tree import (
    CopyPrivate,
    For,
    Name,
    ParallelFor,
    ParallelSections,
    Private,
    PyName,
    Section,
    Sections,
    Single,
    Span,
)
from omp4py.core.preprocessor.transformers.operators import get_reduction, resolve_names
from omp4py.core.preprocessor.transformers.scopes import check_scopes, get_scopes, modify_scope, dec_annotation
from omp4py.core.preprocessor.transformers.symtable import SymbolTable, new_omp_uname, runtime_ast, new_omp_name
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx
from omp4py.core.preprocessor.transformers.utils import fix_body_locations, unpack_if, walk

__all__ = []


@construct.register
def _(ctr: ParallelFor, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    unpack: ast.If = unpack_if(body)
    body = construct(ctr.parallel, [unpack], ctx)
    old_symtable, ctx.symtable = ctx.symtable, ctx.symtable.new_child()
    unpack.body = construct(ctr.for_, unpack.body, ctx)
    ctx.symtable = old_symtable
    return body


@construct.register
def _(ctr: For, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    collapse_num = ctr.collapse.num.value if ctr.collapse is not None else 1
    for_list: list[ast.For] = for_checks(ctr, body, ctx)
    first_private = ctr.first_private
    private = ctr.private

    bounds_name: str = new_omp_uname(ctx, "bounds")
    bounds_value: list[ast.expr]
    bounds_ast: ast.AnnAssign = ctr.span.to_ast(
        ast.AnnAssign(
            ast.Name(bounds_name, ast.Store()),
            ast.Subscript(runtime_ast("pyint"), ast.Constant(3 + 6 * collapse_num)),
            ast.List(bounds_value := [ast.Constant(0) for _ in range(3 + 6 * collapse_num)]),
            simple=1,
        ),
    )

    for_init: ast.stmt = ctr.span.to_ast(ast.Expr(for_initc := ast.Call(runtime_ast("for_init"))))
    for_next: ast.stmt = ctr.span.to_ast(ast.While(for_nextc := ast.Call(runtime_ast("for_next")), body))
    for_end: ast.stmt = ctr.span.to_ast(ast.Expr(for_endc := ast.Call(runtime_ast("for_end"))))
    body = [bounds_ast, for_init, for_next, for_end]

    # Collapse, bounds, modifier, kind, chunk, ordered
    for_initc.args.append(ast.Constant(collapse_num))
    for_initc.args.append(ast.Name(bounds_name))
    for_initc.args.append(ast.Constant(ord(ctr.schedule.type.nkind.string[0] if ctr.schedule else "s")))
    for_initc.args.append(ctr.schedule.chunk.value if ctr.schedule and ctr.schedule.chunk else ast.Constant(-1))
    for_initc.args.append(ast.Constant(0 if not ctr.ordered else (ctr.ordered.n.value if ctr.ordered.n else 1)))

    for_nextc.args.append(ast.Name(bounds_name))

    if ctr.no_wait is None:
        for_endc.args.append(ast.Constant("parallel" in ctr.name.string.lower()))
    elif ctr.no_wait.expr is None:
        for_endc.args.append(ast.Constant(True))
    else:
        for_endc.args.append(ctr.no_wait.expr.value)

    if ctr.last_private:
        check_scopes(ctx, *ctr.last_private)
        all_private_names: set[str] = get_scopes(*ctr.private, *ctr.first_private)
        for scope in ctr.last_private:
            targets: list[PyName] = [name for name in scope.targets if name.string not in all_private_names]
            if targets:
                private.append(Private(scope.span, scope.name, targets))

    for for_ in for_list:
        match for_.target:
            case ast.Name(name):
                if ctx.symtable.get(name, True):
                    span = Span.from_ast(for_.target)
                    private.append(Private(span, Name(span, name), [PyName(span, name)]))

    f_table: SymbolTable = modify_scope(ctr, ctx, body, private, first_private, ctr.reduction)
    for scope in ctr.last_private:
        for var in scope.str_targets:
            if (s := f_table.get(var)) and s.assigned:
                for_list[0].orelse.append(
                    scope.span.to_ast(ast.Assign([ast.Name(s.old_name, ast.Store())], ast.Name(s.scope_name))),
                )

    for i in range(collapse_num):
        for_iter: ast.expr = for_list[i].iter
        new_start: ast.expr = ast.Subscript(ast.Name(bounds_name), ast.Constant(3 + 6 * i))
        new_stop: ast.expr = ast.Subscript(ast.Name(bounds_name), ast.Constant(3 + 6 * i + 1))
        match for_iter:
            case ast.Call(func=ast.Name(id="range"), args=[stop]):
                for_iter.args[0] = new_start
                for_iter.args.append(new_stop)
                bounds_value[3 + 6 * i + 1] = stop
                bounds_value[3 + 6 * i + 2] = ast.Constant(1)
            case ast.Call(func=ast.Name(id="range"), args=[start, stop]):
                for_iter.args[0] = new_start
                for_iter.args[1] = new_stop
                bounds_value[3 + 6 * i] = start
                bounds_value[3 + 6 * i + 1] = stop
                bounds_value[3 + 6 * i + 2] = ast.Constant(1)
            case ast.Call(func=ast.Name(id="range"), args=[start, stop, step]):
                for_iter.args[0] = new_start
                for_iter.args[1] = new_stop
                bounds_value[3 + 6 * i] = start
                bounds_value[3 + 6 * i + 1] = stop
                bounds_value[3 + 6 * i + 2] = step
            case _:
                pass

    return fix_body_locations(body)


def for_range_check(for_: ast.For, ctx: Context, index: list[str] | None = None) -> None:
    if len(for_.orelse) > 0:
        msg = "loop cannot have an else block"
        raise syntax_error_ctx(msg, Span.from_ast(for_.orelse[0]), ctx)

    if not isinstance(for_.target, ast.Name):
        msg = "invalid expression"
        raise syntax_error_ctx(msg, Span.from_ast(for_.target), ctx)

    match for_.iter:
        case ast.Call(func=ast.Name(id="range")):
            if index is not None and len(index) > 0:
                for node in ast.walk(for_.iter):
                    if isinstance(node, ast.Name) and node.id in index:
                        msg = "expression refers to iteration variable"
                        raise syntax_error_ctx(msg, Span.from_ast(node), ctx)

        case _:
            msg = "invalid controlling predicate"
            raise syntax_error_ctx(msg, Span.from_ast(for_.iter), ctx)

    if index is not None:
        index.append(for_.target.id)

    for node in walk(
        for_,
        lambda c: c == for_ or not isinstance(c, (ast.For, ast.AsyncFor, ast.FunctionDef, ast.AsyncFunctionDef)),
    ):
        if isinstance(node, (ast.Break, ast.Return, ast.Yield)):
            msg = f"'{type(node).__name__.lower()}' statement used with OpenMP for loop"
            raise syntax_error_ctx(msg, Span.from_ast(node), ctx)


def for_checks(ctr: For, body: list[ast.stmt], ctx: Context) -> list[ast.For]:
    for_list: list[ast.For] = []

    if len(body) == 0 or not isinstance(stmt := body[0], ast.For):
        msg = "for statement expected"
        span: Span = ctr.name.span if len(body) == 0 else Span.from_ast(body[0])
        raise syntax_error_ctx(msg, span, ctx)
    for_range_check(stmt, ctx)
    for_list.append(stmt)

    if len(body) > 1:
        msg = "unindent expected, but statement found"
        raise syntax_error_ctx(msg, Span.from_ast(body[1]), ctx)

    if ctr.collapse is not None:
        if ctr.collapse.num.value < 1:
            msg = "collapse argument needs positive constant integer expression"
            raise syntax_error_ctx(msg, ctr.collapse.num.span, ctx)

        inner: ast.For = stmt
        index: list[str] = []
        for _ in range(ctr.collapse.num.value - 1):
            if not isinstance(stmt := inner.body[0], ast.For):
                msg = "not enough perfectly nested loops before statement"
                raise syntax_error_ctx(msg, Span.from_ast(stmt), ctx)

            for_range_check(stmt, ctx, index)
            for_list.append(stmt)

            if len(inner.body) > 1:
                msg = "collapsed loops not perfectly nested"
                raise syntax_error_ctx(msg, Span.from_ast(inner.body[1]), ctx)
            inner = stmt

    return for_list


@construct.register
def _(ctr: ParallelSections, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    unpack: ast.If = unpack_if(body)
    body = construct(ctr.parallel, [unpack], ctx)
    old_symtable, ctx.symtable = ctx.symtable, ctx.symtable.new_child()
    unpack.body = construct(ctr.sections, unpack.body, ctx)
    ctx.symtable = old_symtable
    return body


@construct.register
def _(ctr: Section, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    msg = "'omp section' may only be used in 'omp sections' construct"
    raise syntax_error_ctx(msg, ctr.span, ctx)


def section_checks(ctr: Sections, body: list[ast.stmt], ctx: Context) -> list[ast.With]:
    sections_list: list[ast.With] = []
    for i, child in enumerate(body):
        section: ast.With
        if isinstance(child, ast.With) and (d := ctx.directives.get(child)) and isinstance(d.construct, Section):
            section = child
        elif i == 0:
            section = ast.With([ast.withitem(ast.Constant(True))], [child])
        else:
            msg = "expected 'omp section'"
            raise syntax_error_ctx(msg, Span.from_ast(child), ctx)
        sections_list.append(section)
    return sections_list


@construct.register
def _(ctr: Sections, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    sections_list = section_checks(ctr, body, ctx)
    first_private = ctr.first_private
    private = ctr.private

    sec_id: str = new_omp_uname(ctx, "sections_id")
    sec_id_ann: ast.stmt = ctr.span.to_ast(ast.AnnAssign(ast.Name(sec_id, ast.Store()), runtime_ast("pyint"), simple=1))
    sec_init: ast.stmt = ctr.span.to_ast(ast.Expr(sec_initc := ast.Call(runtime_ast("sections_init"))))
    sec_next: ast.stmt = ctr.span.to_ast(sec_loop := ast.While(sec_nextc := ast.Call(runtime_ast("sections_next"))))
    sec_end: ast.stmt = ctr.span.to_ast(ast.Expr(sec_endc := ast.Call(runtime_ast("sections_end"))))

    body = [sec_id_ann, sec_init, sec_next, sec_end]

    sec_initc.args.append(ast.Constant(len(sections_list)))
    sec_loop.test = ast.NamedExpr(ast.Name(sec_id, ast.Store()), sec_nextc)

    for i in range(len(sections_list)):
        sec_loop.body.append(
            ast.copy_location(
                ast.If(ast.Compare(ast.Name(sec_id), [ast.Eq()], [ast.Constant(i)]), sections_list[i].body),
                sections_list[i],
            ),
        )

    if ctr.no_wait is None:
        sec_endc.args.append(ast.Constant("parallel" in ctr.name.string.lower()))
    elif ctr.no_wait.expr is None:
        sec_endc.args.append(ast.Constant(True))
    else:
        sec_endc.args.append(ctr.no_wait.expr.value)

    if ctr.last_private:
        check_scopes(ctx, *ctr.last_private)
        all_private_names: set[str] = get_scopes(*ctr.private, *ctr.first_private)
        for scope in ctr.last_private:
            targets: list[PyName] = [name for name in scope.targets if name.string not in all_private_names]
            if targets:
                private.append(Private(scope.span, scope.name, targets))

    f_table: SymbolTable = modify_scope(ctr, ctx, body, private, first_private, ctr.reduction)
    for scope in ctr.last_private:
        for var in scope.str_targets:
            if (s := f_table.get(var)) and s.assigned:
                sections_list[-1].body.append(
                    scope.span.to_ast(ast.Assign([ast.Name(s.old_name, ast.Store())], ast.Name(s.scope_name))),
                )

    return fix_body_locations(body)


@construct.register
def _(ctr: Single, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    modify_scope(ctr, ctx, body, ctr.private, ctr.first_private, [])

    single_ast = ctr.span.to_ast(ast.If(ast.Call(runtime_ast("single")), body))
    single_end: ast.stmt = ctr.span.to_ast(ast.Expr(single_endc := ast.Call(runtime_ast("single_end"))))

    if ctr.no_wait and ctr.copyprivate:
        msg = "'nowait' clause must not be used together with 'copyprivate'"
        raise syntax_error_ctx(msg, ctr.copyprivate[0].span, ctx)

    if ctr.no_wait is None:
        single_endc.args.append(ast.Constant(False))
    elif ctr.no_wait.expr is None:
        single_endc.args.append(ast.Constant(True))
    else:
        single_endc.args.append(ctr.no_wait.expr.value)

    body: list[ast.stmt] = [single_ast]
    if ctr.copyprivate:
        body.extend(copyprivate(ctr.copyprivate, ctx))
    body.append(single_end)

    return fix_body_locations(body)


def copyprivate(dataclauses: list[CopyPrivate], ctx: Context) -> list[ast.stmt]:
    check_scopes(ctx, *dataclauses, allow_threadprivate=True)
    span = dataclauses[0].span

    return_types: list[ast.expr] = []
    return_names: list[ast.expr] = []
    var_names: list[ast.expr] = []
    types_ast = ast.Subscript(ast.Name("tuple"), ast.Tuple(return_types))
    copyprivate_name = new_omp_uname(ctx, "copyprivate")
    copyprivate_ast = span.to_ast(ast.FunctionDef(copyprivate_name, ast.arguments(), returns=types_ast))

    for dataclause in dataclauses:
        for var in dataclause.targets:
            if s := ctx.symtable.get(var.string, True, True, True):
                stmt = get_reduction(ctx, "__new__", s.annotation)
                if stmt is None:  # never for __new__
                    continue
                new_name = new_omp_name(ctx, s.scope_name)
                assign = dec_annotation(ctx, s, new_name)
                copyprivate_ast.body.append(assign)
                copyprivate_ast.body.append(resolve_names(stmt[1], omp_out=new_name, omp_in=s.scope_name))

                var_names.append(ast.Name(s.scope_name, ast.Store()))
                return_names.append(ast.Name(new_name))
                return_types.append(assign.annotation)

    copyprivate_ast.body.append(ast.Return(ast.Tuple(return_names)))

    copy_wait_ast: ast.stmt = span.to_ast(
        ast.AnnAssign(
            ast.Name(copyprivate_name, ast.Store()),
            ast.Call(runtime_ast("cy_typeof"), [ast.Name(copyprivate_name)]),
            ast.Call(runtime_ast("copyprivate_wait"), [ast.Name(copyprivate_name)]),
            simple=1,
        ),
    )
    update_ast = span.to_ast(
        ast.Assign(
            [ast.Tuple(var_names, ast.Store())], ast.Call(ast.Name(copyprivate_name), [ast.Name(copyprivate_name)], [])
        ),
    )

    return [copyprivate_ast, copy_wait_ast, update_ast]
