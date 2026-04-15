"""Data environment transformation module.

This module implements the `threadprivate` OpenMP-like construct.

The goal of this transformation is to ensure that each thread has its
own independent instance of selected global variables, while preserving
their logical identity across the program.

The transformation performs:
- Validation of threadprivate variables (must be global scope)
- Registration of threadprivate symbols in module storage
- Renaming and rewriting of identifiers in the AST
- Replacement of variable accesses with runtime thread-local access
- Prevention of unsafe usage patterns (e.g. named expressions)
- Deferred AST rewriting via finalizers in the transformation context
"""

from __future__ import annotations

import ast
import copy

from omp4py.core.parser.tree import Span, ThreadPrivate
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx

__all__ = []

@construct.register
def _(ctr: ThreadPrivate, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]: # noqa: C901
    """Transform the OpenMP-like `threadprivate` construct into a runtime-aware AST representation.

    This transformation implements thread-local storage semantics for global
    variables, ensuring that each thread operates on an independent instance
    of the variable while preserving a consistent logical identifier.

    The process performs the following steps:

    1. Validation:
       - Ensures each declared variable exists in the current symbol table.
       - Verifies that the variable is globally scoped (required by semantics).

    2. Registration:
       - Marks the variable as threadprivate in module-level storage.
       - Ensures the symbol exists in the current scope if previously undefined.

    3. Renaming:
       - Introduces a renamed store target to avoid naming conflicts.
       - Updates the symbol table with a renamed binding.
       - Tracks mapping between original and transformed names.

    4. Initialization emission:
       - Generates an `AnnAssign` node that initializes threadprivate storage
         using the runtime helper `threadprivate(name, value)`.

    5. Deferred rewriting:
       - Registers a finalizer that rewrites all AST occurrences of the
         variable in the enclosing scope.
       - Replaces variable accesses with runtime calls:
           threadprivates(name).v
       - Applies optional type casting using `cy_cast` if annotations exist.
       - Prevents unsafe constructs such as assignment expressions (`:=`).

    Args:
        ctr (ThreadPrivate): Parsed threadprivate directive containing targets.
        body (list[ast.stmt]): Body of the construct (not modified directly).
        ctx (Context): Current transformation context containing symbol table,
            scope information, and module-level storage.

    Returns:
        list[ast.stmt]: A list of AST statements consisting of initialization
        code followed by the original body of the construct.
    """
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
        if ctx.symtable.get(var.string) is None:
            ctx.symtable.update(ast.Name(var.string))
            s = ctx.symtable[var.string]

        target = ast.Name(var.string, ast.Store())
        ctx.symtable.rename({var.string}, target)
        new_names[var.string] = target.id
        s.threadprivate = True
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

    # AST transformer used for deferred rewriting of threadprivate accesses
    class Replace(ast.NodeTransformer):
        def visit_TypeAlias(self, node: ast.TypeAlias) -> ast.TypeAlias:
            return node

        def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.NamedExpr:
            # Disallow assignment expressions involving threadprivate variables
            if node.target.id not in new_names or node.lineno < ctr.span.lineno:
                return node
            msg = "cannot use assignment expressions with threadprivate"
            raise syntax_error_ctx(msg, Span.from_ast(node), ctx)

        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id not in new_names or node.lineno <= ctr.span.lineno:
                return node
            new_node = self.replaced(node)

            return ast.fix_missing_locations(ast.copy_location(new_node, node))

        def replaced(self, node: ast.Name) -> ast.Attribute | ast.Call:
            # Replace variable access with runtime thread-local access
            new_node = ast.Attribute(
                ast.Call(runtime_ast("threadprivates"), [ast.Name(new_names[node.id])]),
                "v",
                node.ctx,
            )

            if isinstance(node.ctx, ast.Load):
                ann: ast.expr | None = copy.deepcopy(ctx.module_storage.threadprivate[node.id].annotation)
                if ann:
                    new_node = ast.Call(runtime_ast("cy_cast"), [ann, new_node])

            return new_node

    # Schedule transformation to be applied after full traversal
    scope_node = ctx.scope.node
    ctx.finalizers.append(lambda:Replace().visit(scope_node))

    return tp_vars + body
