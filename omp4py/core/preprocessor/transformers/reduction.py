import ast

from omp4py.core.parser.tree import DeclareReduction
from omp4py.core.preprocessor.transformers.operators import new_reduction
from omp4py.core.preprocessor.transformers.transformer import Context, construct, syntax_error_ctx


@construct.register
def _(ctr: DeclareReduction, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    init: ast.stmt | None = ctr.initializer.stmt.value if ctr.initializer else None
    comb: ast.stmt = ctr.combiner.stmt.value

    for ann in ctr.ann_list:
        if not new_reduction(ctx, ctr.id.value, ann.value, comb, init):
            msg = f"redeclaration of '{ctr.id.value}' 'omp declare reduction' for type '{ast.unparse(ann.value)}'"
            raise syntax_error_ctx(msg, ann.span, ctx)

    return body
