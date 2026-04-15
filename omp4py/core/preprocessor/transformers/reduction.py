"""Reduction declaration construct for the `omp4py` preprocessor.

This module implements the OpenMP `declare reduction` construct, which
allows users to define custom reduction operations.

A declared reduction specifies:

- A combiner: how partial results are merged
- An optional initializer: how private copies are initialized
- A type or set of types for which the reduction applies

Additionally, the special reduction operation `__new__` can be defined
to control how values of a given type are initialized and copied. This
mechanism is reused across the system to provide consistent object
creation and copy semantics, not only for reductions but also for other
data-sharing constructs.

The construct registers reduction operations in the module-level storage,
making them available for use in `reduction` clauses during transformation.

All constructs are registered through the `construct` dispatcher and
applied during the transformation phase of the preprocessor.
"""

from __future__ import annotations

import ast
import typing

from omp4py.core.parser.tree import DeclareReduction  # noqa: TC001 required
from omp4py.core.preprocessor.transformers.operators import new_reduction
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx

__all__ = []


@construct.register
def _(ctr: DeclareReduction, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Transform an OpenMP `declare reduction` construct.

    Registers a user-defined reduction operation in the current module
    context. The reduction is defined by a combiner statement and an
    optional initializer, and is associated with one or more types.

    If a reduction with the same name and type is already defined, a
    syntax error is raised.

    Args:
        ctr (DeclareReduction): Parsed `declare reduction` construct.
        body (list[ast.stmt]): Body of the construct (typically unused).
        ctx (Context): Transformation context.

    Returns:
        list[ast.stmt]: The original body, unchanged.

    Raises:
        SyntaxError: If the reduction is redeclared for a given type.
    """
    init: ast.stmt | None = ctr.initializer.stmt.value if ctr.initializer else None
    comb: ast.stmt = ctr.combiner.stmt.value

    for ann in ctr.ann_list:
        if not new_reduction(ctx, ctr.id.value, ann.value, comb, init):
            msg = f"redeclaration of '{ctr.id.value}' 'omp declare reduction' for type '{ast.unparse(ann.value)}'"
            raise syntax_error_ctx(msg, ann.span, ctx)

    return body
