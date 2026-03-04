import ast
import copy

from omp4py.core.parser.tree import Span, ThreadPrivate
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx


@construct.register
def _(ctr: ThreadPrivate, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    tp_vars: list[ast.stmt] = []
    new_names: dict[str, str] = {}
    for var in ctr.targets:
        if not (s := ctx.symtable.get(var.string, True, True)):
            msg = f"name '{var.string}' is not defined"
            raise syntax_error_ctx(msg, var.span, ctx)

        if not s.global_:
            msg = f"local variable '{var.string}' cannot be 'threadprivate'"
            raise syntax_error_ctx(msg, var.span, ctx)

        ctx.module_storage.threadprivate[var.string] = s
        target = ast.Name(var.string, ast.Store())
        ctx.symtable.rename({var.string}, target)
        new_names[var.string] = target.id
        tp_vars.append(
            ast.fix_missing_locations(
                var.span.to_ast(
                    ast.AnnAssign(
                        target=target,
                        annotation=runtime_ast("pyint"),
                        value=ast.Call(runtime_ast("threadprivate"), [ast.Constant(var.string), ast.Name(var.string)]),
                        simple=1,
                    ),
                ),
            ),
        )

    class Replace(ast.NodeTransformer):
        def visit_TypeAlias(self, node: ast.TypeAlias) -> ast.TypeAlias:
            return node

        def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.NamedExpr:
            if node.target.id not in new_names or node.lineno < ctr.span.lineno:
                return node
            msg = "cannot use assignment expressions with threadprivate"
            raise syntax_error_ctx(msg, Span.from_ast(node), ctx)

        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id not in new_names or node.lineno < ctr.span.lineno:
                return node
            new_node = self.replaced(node)

            return ast.fix_missing_locations(ast.copy_location(new_node, node))

        def replaced(self, node: ast.Name) -> ast.Subscript | ast.Call:
            new_node = ast.Subscript(
                ast.Call(runtime_ast("threadprivates")),
                ast.Name(new_names[node.id]),
                node.ctx,
            )

            if isinstance(node.ctx, ast.Load):
                ann: ast.expr | None = copy.deepcopy(ctx.module_storage.threadprivate[node.id].annotation)
                if ann:
                    new_node = ast.Call(runtime_ast("cast"), [ann, new_node])

            return new_node

    Replace().visit(ctx.scope_node)

    return tp_vars + body
