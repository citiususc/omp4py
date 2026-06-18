from __future__ import annotations

import ast as pyast
from typing import cast

from . import tree
from .openmp_parser import Transformer, v_args, Token, Tree, Meta
from .source_view import SourceView

__all__ = ["AstTransformer"]

@v_args(tree=True)
class AstTransformer(Transformer):

    def __init__(self, sv: SourceView) -> None:
        super().__init__()
        self.sv = sv

    #### HELPERS ###############################################################

    def _token2span(self, token: Token) -> tree.Span:
        if token.line is None or token.column is None:
            msg = "Missing position information"
            raise ValueError(msg)

        # Lark starts at line 1 column 1, but the Span expects an offset.
        return tree.Span(
            token.line,
            token.column - 1,
            token.end_line       if token.end_line   is not None else -1,
            token.end_column - 1 if token.end_column is not None else -1,
        )

    def _meta2span(self, meta: Meta) -> tree.Span:
        if meta.empty:
            msg = "Meta object is empty"
            raise ValueError(msg)

        # Lark starts at line 1 column 1, but the Span expects an offset.
        return tree.Span(
            meta.line,
            meta.column - 1,
            meta.end_line,
            meta.end_column - 1,
        )

    # No type here, otherwise cast() will be required everywhere
    def _name_from_token(self, token) -> tree.Name:
        return tree.Name(span=self._token2span(token), string=str(token))

    # Applies to rules like: name_clause: KEYWORD_CLAUSE "(" var_list ")"
    def _fill_data_scope[T: tree.DataScope](
        self,
        node: Tree,
        cls: type[T],
        target_child: int=1,
        **kwargs
    ) -> T:
        return cls(
            span    = self._meta2span(node.meta),
            name    = self._name_from_token(node.children[0]),
            targets = cast("list[tree.PyName]", node.children[target_child]),
            **kwargs,
        )

    def _set_clause(self, construct: tree.Construct, clause: tree.Clause) -> bool:
        if not hasattr(construct, clause.id):
            return False

        current = getattr(construct, clause.id, None)

        # If the construct's field hasn't been set, update it
        if current is None:
            setattr(construct, clause.id, clause)
            return True

        # If the construct's field is a list, append the new clause
        elif isinstance(current, list):
            current.append(clause)
            return True

        # Otherwise, means that the construct's field has already been set,
        # so the clause is duplicated. Therefore, raise an error.
        else:
            raise self.sv.syntax_error(
                f"{clause.id} clause can only be defined once.",
                clause.span,
                diagnostics=[("first defined here", current.span)],
            )


    # Applies to rules like: KEYWORD _rule_clause_list?
    def _fill_construct[T: tree.Construct](self, node: Tree, cls: type[T]) -> T:
        construct = cls(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
        )

        for clause in node.children[1:]:
            clause = cast("tree.Clause", clause)
            if not self._set_clause(construct, clause):
                msg = f"{construct.__class__.__name__} does not accept clause '{clause.id}'"
                raise TypeError(msg)

        return construct

    #### TOKENS ################################################################

    # IDENTIFIER: /[^\W\d]\w*/
    @v_args(inline=True)
    def IDENTIFIER(self, token: Token) -> tree.PyName:
        span = self._token2span(token)

        # This ensures that the token is considered an identifier by Python
        if not token.value.isidentifier():
            raise self.sv.syntax_error("invalid characters found in identifier", span)

        return tree.PyName(span=span, string=token.value)

    # INTEGER: /[0-9]+/
    @v_args(inline=True)
    def INTEGER(self, token: Token) -> tree.PyInt:
        # The int() conversion is safe to do here because the parser guarantees only digits
        # Also, if the base is 0, Python will correctly guess it based on the prefix:
        #     XXX    ==> Base 10
        #     0bXXXX ==> Base 2
        #     0oXXX  ==> Base 8
        #     0xXX   ==> Base 16
        return tree.PyInt(span=self._token2span(token), value=int(token, 0))

    # py_expr: (PY_ATOM | "(" py_expr ")")+
    @v_args(tree=True)
    def py_expr(self, node: Tree) -> tree.PyExpr:
        try:
            # Get the source of the expression and pass it to ast.parse()
            span = self._meta2span(node.meta)
            source = self.sv.directive_text(span)

            # Clean up leading whitespace, as it will be treated as indentation
            source_stripped = source.lstrip()
            leading_ws = len(source) - len(source_stripped)

            parsed = pyast.parse(source_stripped, mode="eval")
            return tree.PyExpr(
                span=span,
                value=parsed.body,
                source=source,
            )

        except SyntaxError as e:
            # Adjust the error position to show it within its context
            err_line     = e.lineno     or 1
            err_end_line = e.end_lineno or err_line
            err_col      = e.offset     or 1
            err_end_col  = e.end_offset or err_col
            span = tree.Span(
                *self.sv.absolute_position(span, err_line,     err_col,     leading_ws),
                *self.sv.absolute_position(span, err_end_line, err_end_col, leading_ws),
            )

            msg = f"invalid Python expression: {e.msg}"
            raise self.sv.syntax_error(msg, span) from None

    #### MODIFIERS #############################################################

    # var_list: IDENTIFIER ("," IDENTIFIER)*
    @v_args(inline=True)
    def var_list(self, *names: tree.PyName) -> list[tree.PyName]:
        return list(names)

    # reduction_op: PLUS | MINUS | MULT | BITWISE_AND | BITWISE_OR | BITWISE_XOR | LOGIC_AND | LOGIC_OR | MAX | MIN
    @v_args(inline=True)
    def reduction_op(self, token: Token) -> tree.ReductionOp:
        return tree.ReductionOp(span=self._token2span(token), value=str(token))

    # schedule_type: STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
    @v_args(inline=True)
    def schedule_type(self, token: Token) -> tree.ScheduleType:
        span = self._token2span(token)
        name = tree.Name(span=span, string=str(token))
        return tree.ScheduleType(span=span, nkind=name)

    #### CLAUSES ###############################################################

    # shared_clause: SHARED_CLAUSE "(" var_list ")"
    def shared_clause(self, node: Tree) -> tree.Shared:
        return self._fill_data_scope(node, tree.Shared)

    # private_clause: PRIVATE_CLAUSE "(" var_list ")"
    def private_clause(self, node: Tree) -> tree.Private:
        return self._fill_data_scope(node, tree.Private)

    # firstprivate_clause: FIRSTPRIVATE_CLAUSE "(" var_list ")"
    def firstprivate_clause(self, node: Tree) -> tree.FirstPrivate:
        return self._fill_data_scope(node, tree.FirstPrivate)

    # lastprivate_clause: LASTPRIVATE_CLAUSE "(" var_list ")"
    def lastprivate_clause(self, node: Tree) -> tree.LastPrivate:
        return self._fill_data_scope(node, tree.LastPrivate)

    # copyin_clause: COPYIN_CLAUSE "(" var_list ")"
    def copyin_clause(self, node: Tree) -> tree.CopyIn:
        return self._fill_data_scope(node, tree.CopyIn)

    # copyprivate_clause: COPYPRIVATE_CLAUSE "(" var_list ")"
    def copyprivate_clause(self, node: Tree) -> tree.CopyPrivate:
        return self._fill_data_scope(node, tree.CopyPrivate)

    # reduction_clause: REDUCTION_CLAUSE "(" reduction_op ":" var_list ")"
    def reduction_clause(self, node: Tree) -> tree.Reduction:
        return self._fill_data_scope(node, tree.Reduction, target_child=2, op=node.children[1])


    # schedule_clause: SCHEDULE_CLAUSE "(" schedule_type ("," py_expr)? ")"
    def schedule_clause(self, node: Tree) -> tree.Schedule:
        return tree.Schedule(
            span  = self._meta2span(node.meta),
            name  = self._name_from_token(node.children[0]),
            type  = cast("tree.ScheduleType", node.children[1]),
            chunk = cast("tree.PyExpr",       node.children[2]) if len(node.children) > 2 else None,
        )

    # default_clause: DEFAULT_CLAUSE "(" (SHARED | NONE) ")"
    def default_clause(self, node: Tree) -> tree.Default:
        return tree.Default(
            span  = self._meta2span(node.meta),
            name  = self._name_from_token(node.children[0]),
            ntype = self._name_from_token(node.children[1])
        )

    # if_clause: IF_CLAUSE "(" py_expr ")"
    def if_clause(self, node: Tree) -> tree.If:
        return tree.If(
            span = self._meta2span(node.meta),
            name = self._name_from_token(node.children[0]),
            expr = cast("tree.PyExpr", node.children[1]),
        )

    # num_threads_clause: NUM_THREADS_CLAUSE "(" py_expr ")"
    def num_threads_clause(self, node: Tree) -> tree.NumThreads:
        return tree.NumThreads(
            span = self._meta2span(node.meta),
            name = self._name_from_token(node.children[0]),
            expr = cast("tree.PyExpr", node.children[1])
        )

    # final_clause: FINAL_CLAUSE "(" py_expr ")"
    def final_clause(self, node: Tree) -> tree.Final:
        return tree.Final(
            span = self._meta2span(node.meta),
            name = self._name_from_token(node.children[0]),
            expr = cast("tree.PyExpr", node.children[1])
        )

    # collapse_clause: COLLAPSE_CLAUSE "(" INTEGER ")"
    def collapse_clause(self, node: Tree) -> tree.Collapse:
        num = cast("tree.PyInt", node.children[1])
        if num.value <= 0:
            raise self.sv.syntax_error("required positive non-zero integer.", num.span)
        return tree.Collapse(
            span = self._meta2span(node.meta),
            name = self._name_from_token(node.children[0]),
            num  = num,
        )

    # ordered_clause: ORDERED_CLAUSE
    def ordered_clause(self, node: Tree) -> tree.OrderedClause:
        # TODO: tree.OrderedClause.n
        return tree.OrderedClause(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    # nowait_clause: NOWAIT_CLAUSE
    def nowait_clause(self, node: Tree) -> tree.NoWait:
        # TODO: tree.NoWait.expr
        return tree.NoWait(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            expr=None
        )

    # untied_clause: UNTIED_CLAUSE
    def untied_clause(self, node: Tree) -> tree.Untied:
        return tree.Untied(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    # mergeable_clause: MERGEABLE_CLAUSE
    def mergeable_clause(self, node: Tree) -> tree.Mergeable:
        return tree.Mergeable(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    #### CONSTRUCTS ############################################################

    def parallel_directive(self, node: Tree) -> tree.Parallel:
        return self._fill_construct(node, tree.Parallel)

    def parallel_for_directive(self, node: Tree) -> tree.ParallelFor:
        parallel_name = self._name_from_token(node.children[0])
        for_name      = self._name_from_token(node.children[1])

        # Combine both spans
        parallel_for_span = tree.Span(
            parallel_name.span.lineno,
            parallel_name.span.offset,
            for_name.span.end_lineno,
            for_name.span.end_offset,
        )
        parallel_for_name = tree.Name(
            parallel_for_span,
            self.sv.directive_text(parallel_for_span),
        )

        construct_span = self._meta2span(node.meta)
        parallel = tree.Parallel(construct_span, parallel_name)
        for_ = tree.For(construct_span, for_name)

        for clause in node.children[2:]:
            clause = cast("tree.Clause", clause)
            set_parallel_clause = self._set_clause(parallel, clause)
            set_for_clause      = self._set_clause(for_, clause)

            if not set_parallel_clause and not set_for_clause:
                msg = f"{parallel.__class__.__name__} nor {for_.__class__.__name__} do not accept clause '{clause.id}'"
                raise TypeError(msg)

        return tree.ParallelFor(
            construct_span,
            parallel_for_name,
            parallel,
            for_
        )

    def parallel_sections_directive(self, node: Tree) -> tree.ParallelSections:
        parallel_name = self._name_from_token(node.children[0])
        sections_name = self._name_from_token(node.children[1])

        # Combine both spans
        parallel_sections_span = tree.Span(
            parallel_name.span.lineno,
            parallel_name.span.offset,
            sections_name.span.end_lineno,
            sections_name.span.end_offset,
        )
        parallel_sections_name = tree.Name(
            parallel_sections_span,
            self.sv.directive_text(parallel_sections_span),
        )

        construct_span = self._meta2span(node.meta)
        parallel = tree.Parallel(construct_span, parallel_name)
        sections = tree.Sections(construct_span, sections_name)

        for clause in node.children[2:]:
            clause = cast("tree.Clause", clause)
            set_parallel_clause = self._set_clause(parallel, clause)
            set_sections_clause = self._set_clause(sections, clause)

            if not set_parallel_clause and not set_sections_clause:
                msg = f"{parallel.__class__.__name__} nor {sections.__class__.__name__} do not accept clause '{clause.id}'"
                raise TypeError(msg)

        return tree.ParallelSections(
            construct_span,
            parallel_sections_name,
            parallel,
            sections,
        )

    def for_directive(self, node: Tree) -> tree.For:
        return self._fill_construct(node, tree.For)

    def sections_directive(self, node: Tree) -> tree.Sections:
        return self._fill_construct(node, tree.Sections)

    def section_directive(self, node: Tree) -> tree.Section:
        return tree.Section(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    def single_directive(self, node: Tree) -> tree.Single:
        return self._fill_construct(node, tree.Single)

    def task_directive(self, node: Tree) -> tree.Task:
        return self._fill_construct(node, tree.Task)


    def master_directive(self, node: Tree) -> tree.Master:
        return tree.Master(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    # TODO: handle critical identifier
    # critical_directive: CRITICAL ("(" IDENTIFIER ")")?
    def critical_directive(self, node: Tree) -> tree.Critical:
        return tree.Critical(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    def barrier_directive(self, node: Tree) -> tree.Barrier:
        return tree.Barrier(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    def ordered_directive(self, node: Tree) -> tree.Ordered:
        return tree.Ordered(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    # threadprivate_directive: THREADPRIVATE "(" var_list ")"
    def threadprivate_directive(self, node: Tree) -> tree.ThreadPrivate:
        return tree.ThreadPrivate(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            targets=cast("list[tree.PyName]", node.children[1]),
        )

    def taskyield_directive(self, node: Tree) -> tree.TaskYield:
        return tree.TaskYield(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    def taskwait_directive(self, node: Tree) -> tree.TaskWait:
        return tree.TaskWait(
            span=self._meta2span(node.meta),
            name=self._name_from_token(node.children[0])
        )

    def atomic_directive(self, node: Tree) -> tree.Atomic:
        return tree.Atomic(
            span  = self._meta2span(node.meta),
            name  = self._name_from_token(node.children[0]),
            ntype = self._name_from_token(node.children[1]) if len(node.children) > 1 else None,
        )

    def flush_directive(self, node: Tree) -> tree.Flush:
        return tree.Flush(
            span    = self._meta2span(node.meta),
            name    = self._name_from_token(node.children[0]),
            targets = cast("list[tree.PyName]", node.children[1]) if len(node.children) > 1 else None,
        )

    ############################################################################

    @v_args(inline=True)
    def start(self, construct: tree.Construct) -> tree.Directive:
        return tree.Directive(
            span=construct.span,
            string=self.sv.directive,
            construct=construct,
        )
