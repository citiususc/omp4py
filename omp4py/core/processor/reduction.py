import ast
import typing

from omp4py.core.directive import names
from omp4py.core.processor.processor import omp_processor
from omp4py.core.directive import OmpDirective, OmpClause, OmpArgs, OmpItem
from omp4py.core.processor.nodes import NodeContext, check_nobody, ast_search
from omp4py.core.processor.varscope import VariableVisitor
from omp4py.core.processor import common


@omp_processor(names.D_DECLARE_REDUCTION)
def d_reduct(body: list[ast.stmt], clauses: list[OmpClause], args: OmpArgs | None, ctx: NodeContext) -> list[ast.stmt]:
    check_nobody(ctx, body)

    clause: OmpClause
    init: OmpItem = ...
    comb: OmpItem = ...
    for clause in clauses:
        match str(clause):
            case names.C_INITIALIZER:
                init = clause.args.array[0]
            case names.C_COMBINER:
                comb = clause.args.array[0]

    init_vars: set[str] = set().union(*VariableVisitor.search(init.value))
    comb_vars: set[str] = set().union(*VariableVisitor.search(comb.value))

    if len(err := init_vars - {'omp_orig', 'omp_priv'}) > 0:
        var: str = sorted(err)[0]
        raise ctx.error(f" undeclared '{var}'", ast_search(var, init.value))

    if len(err := comb_vars - {'omp_in', 'omp_out'}) > 0:
        var: str = sorted(err)[0]
        raise ctx.error(f" undeclared '{var}'", ast_search(var, init.value))

    return body