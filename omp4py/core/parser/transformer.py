from __future__ import annotations

import ast as pyast
from dataclasses import fields, MISSING
from typing import cast, get_type_hints, get_origin, get_args, Union, Iterable, Any
from types import UnionType

from . import tree
from .openmp_parser import Transformer, v_args, Token, Tree, Meta
from .source_view import SourceView

__all__ = ["AstTransformer"]

# Rule name to tree class look-up table
_DIRECTIVE_TYPES = {
    "PARALLEL_DIRECTIVE": tree.Parallel,
    "FOR_DIRECTIVE": tree.For,
    "SECTIONS_DIRECTIVE": tree.Sections,
    "SECTION_DIRECTIVE": tree.Section,
    "SINGLE_DIRECTIVE": tree.Single,
    "SIMD_DIRECTIVE": tree.Simd,
    "DECLARE_SIMD_DIRECTIVE": tree.DeclareSimd,
    "TASK_DIRECTIVE": tree.Task,
    "TASKLOOP_DIRECTIVE": tree.Taskloop,
    "TASKYIELD_DIRECTIVE": tree.Taskyield,
    "TARGET_DATA_DIRECTIVE": tree.TargetData,
    "TARGET_ENTER_DATA_DIRECTIVE": tree.TargetEnterData,
    "TARGET_EXIT_DATA_DIRECTIVE": tree.TargetExitData,
    "TARGET_DIRECTIVE": tree.Target,
    "TARGET_UPDATE_DIRECTIVE": tree.TargetUpdate,
    "DECLARE_TARGET_DIRECTIVE": tree.DeclareTarget,
    "TEAMS_DIRECTIVE": tree.Teams,
    "DISTRIBUTE_DIRECTIVE": tree.Distribute,
    "MASTER_DIRECTIVE": tree.Master,
    "CRITICAL_DIRECTIVE": tree.Critical,
    "BARRIER_DIRECTIVE": tree.Barrier,
    "TASKWAIT_DIRECTIVE": tree.Taskwait,
    "TASKGROUP_DIRECTIVE": tree.Taskgroup,
    "ATOMIC_DIRECTIVE": tree.Atomic,
    "FLUSH_DIRECTIVE": tree.Flush,
    "ORDERED_DIRECTIVE": tree.Ordered,
    "CANCEL_DIRECTIVE": tree.Cancel,
    "CANCELLATION_POINT_DIRECTIVE": tree.CancellationPoint,
    "THREADPRIVATE_DIRECTIVE": tree.ThreadPrivate,
    "DECLARE_REDUCTION_DIRECTIVE": tree.DeclareReduction,
}

# TODO: possible lru_cache here
def _has_field(field_name: str, cls: type[Any], ignore: set[str]|None = None) -> bool:
    """Checks if the dataclass `cls` contains the field `field_name`,
    excluding any listed in the `ignore` set.
    Note that this function does not check if it is from the same dataclass.
    """
    # avoid mutable default arguments
    if ignore is None:
        ignore = {"span", "name"}

    # check ignore set
    if field_name in ignore:
        return False

    # check if field name is present
    field_names = {f.name for f in fields(cls)}
    return field_name in field_names

# TODO: possible lru_cache here
def _required_fields(cls: type[Any], ignore: set[str] = {"span", "name"}) -> set[str]:
    """Returns the names of the fields of the dataclass `cls`
    that must be supplied when initializing it,
    excluding any fields listed in `ignore`.
    """
    return {
        f.name
        for f in fields(cls)
        if f.init and f.default is MISSING and f.default_factory is MISSING and f.name not in ignore
    }

def _is_clause_type(hint) -> bool:
    """Checks if type annotation represents a `tree.Clause`,
    a list of clauses, or a union containing one of those types.

    Examples:
    ```
    _is_clause_type(tree.Clause)              # True
    _is_clause_type(list[tree.Clause])        # True
    _is_clause_type(Union[tree.Clause, None]) # True
    _is_clause_type(str)                      # False
    _is_clause_type(list[str])                # False
    ```
    """
    origin = get_origin(hint)
    if origin is list:
        args = get_args(hint)
        return bool(args) and isinstance(args[0], type) and issubclass(args[0], tree.Clause)

    if origin is (Union, UnionType):
        args = get_args(hint)
        return any(_is_clause_type(a) for a in args if a is not type(None))

    return isinstance(hint, type) and issubclass(hint, tree.Clause)


@v_args(tree=True)
class AstTransformer(Transformer):

    def __init__(self, sv: SourceView) -> None:
        super().__init__()
        self.sv = sv

    #### HELPERS ###############################################################

    # No type here, otherwise cast() will be required everywhere
    def _name_from_token(self, token) -> tree.Name:
        return tree.Name(span=self.sv.token2span(token), string=str(token))

    #### PYTHON CODE HELPERS ###################################################

    def _parse_py_code(self, meta: Meta) -> tuple[pyast.Module, tree.Span, str]:
        """Retrieves the original code and passes it to `ast.parse()`."""
        try:
            # Get the source of the expression and pass it to ast.parse()
            span = self.sv.meta2span(meta)
            source = self.sv.source_text(span)

            # Clean up leading whitespace, as it will be treated as indentation
            source_stripped = source.lstrip()
            leading_ws = len(source) - len(source_stripped)

            code = pyast.parse(source_stripped)
            code = self._apply_span(code, span, leading_ws)
            return code, span, source

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

            msg = f"invalid Python code: {e.msg}"
            raise self.sv.syntax_error(msg, span) from None

    def _apply_span[T: pyast.AST](self, code: T, span: tree.Span, leading_ws: int = 0) -> T:
        """Modifies the Python AST's positions so they are relative to the whole file."""
        node: pyast.AST
        for node in pyast.walk(code):
            starts_on_first_line = False

            if hasattr(node, "lineno"):
                starts_on_first_line = node.lineno == 1
                node.lineno += span.lineno - 1 # ty:ignore[unsupported-operator] # zuban:ignore[attr-defined]

            if hasattr(node, "end_lineno"):
                ends_on_first_line = node.end_lineno == 1
                node.end_lineno += span.lineno - 1 # ty:ignore[unsupported-operator] # zuban:ignore[attr-defined]
            else:
                # Assume single-line node
                ends_on_first_line = starts_on_first_line

            if starts_on_first_line and hasattr(node, "col_offset"):
                node.col_offset += span.offset + leading_ws # ty:ignore[unsupported-operator] # zuban:ignore[attr-defined]

            if ends_on_first_line and hasattr(node, "end_col_offset"):
                node.end_col_offset += span.offset + leading_ws # ty:ignore[unsupported-operator] # zuban:ignore[attr-defined]
        return code

    #### CLAUSE HELPERS ########################################################

    def _simple_clause[T: tree.Clause](self, cls: type[T], field: str, node: Tree) -> T:
        """Build a clause with one optional modifier and a single argument.

        Extracts the source span and clause name from the parse tree,
        then identifies the optional directive name and argument.
        The argument is stored in the specified field of the resulting clause.

        This helper is intended for simple clause rules
        whose modifier cannot be repeated.
        Use `_clause_with_mods` for clauses with modifier rules.

        Args:
            cls: Clause class to instantiate.
            field: Name of the clause field that receives the argument.
            node: Parse tree node representing the clause.

        Returns:
            An instance of `cls` populated with
            the source span, clause name, optional directive name, and argument.

        Raises:
            AssertionError: If the parse tree has an unexpected structure.

        Example:
            private_clause: PRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"

            This rule can be handled with:
                _simple_clause(tree.Private, "targets", node)
        """
        span = self.sv.meta2span(node.meta)
        name = self._name_from_token(node.children[0])
        directive_name = None

        # Three children indicate a directive name and an argument
        if len(node.children) == 3 and isinstance(node.children[1], tree.DirectiveName):
            directive_name = node.children[1]
            arg = (
                node.children[2].children[0]
                if isinstance(node.children[2], Tree)
                else node.children[2]
            )

        # Two children indicate an argument without a directive name
        elif len(node.children) == 2:
            arg = (
                node.children[1].children[0]
                if isinstance(node.children[1], Tree)
                else node.children[1]
            )

        # A single child represents a clause with no argument
        elif len(node.children) == 1:
            arg = None

        # Reject grammar structures that this helper cannot represent
        else:
            assert False, "_simple_clause() used on a complex clause"

        return cls(span=span, name=name, directive_name=directive_name, **{field: arg})

    # IMPORTANT: don't use when the modifier can be repeated
    def _clause_with_mods[T: tree.Clause](
        self,
        cls: type[T],
        meta: Meta,
        token,
        mod_list: list,
        **kwargs
    ) -> T:
        """Build a clause from a token and a collection of optional modifiers.

        Converts parsed modifier rules into keyword arguments for the clause constructor.
        Validates modifier names, rejects duplicate modifiers,
        and checks that all required constructor fields are provided.

        Modifier rules may be represented by parse-tree nodes
        or existing `tree.Modifier` instances.
        A modifier rule represented by a parse-tree node must contain exactly one child.

        Args:
            cls: Clause class to instantiate.
            meta: Parser metadata used to determine the source span.
            token: Token identifying the clause.
            mod_list: Parsed modifiers, possibly containing absent modifiers represented by None.
            **kwargs: Additional clause fields supplied by the caller.

        Returns:
            An instance of `cls` populated with
            the clause name, source span, modifiers, and additional fields.

        Raises:
            AssertionError: If the token or modifier grammar structure is invalid.
            TypeError: If a modifier has an unsupported representation.
            SyntaxError: If a modifier is duplicated
                or a required field is missing,
                using the syntax-error mechanism provided by `self.sv`.
        """
        span = self.sv.meta2span(meta)
        assert isinstance(token, Token)
        name = self._name_from_token(token)

        for mod in mod_list:
            if mod is None:
                continue

            field_name: str
            # Parse-tree nodes carry the destination field name in their rule
            if isinstance(mod, Tree):
                assert _has_field(mod.data, cls), "Incorrect grammar: rule name does not match field"
                assert len(mod.children) == 1, "Incorrect grammar: modifier rules must be a single token"

                # Use the grammar rule name as the destination field
                field_name = mod.data
                mod = (
                    mod.children[0]
                    if isinstance(mod.children[0], tree.Modifier)
                    else self._name_from_token(mod.children[0])
                )

            # An existing Modifier already identifies its destination field
            elif isinstance(mod, tree.Modifier):
                field_name = mod.id

            # Reject modifier representations this helper does not understand
            else:
                raise TypeError(f"unexpected modifier kind: {mod!r}")

            # A previously supplied value means the same modifier appeared twice
            if field_name in kwargs:
                raise self.sv.syntax_error(
                    f'"{field_name}" modifier is already defined for {cls.id} clause.', span,
                    diagnostics=[("first defined here", kwargs[field_name].span)]
                )
            kwargs[field_name] = mod

        # Before creating the final object, check if the required fields are set
        missing = _required_fields(cls) - kwargs.keys()
        if missing:
            raise self.sv.syntax_error(
                f'missing required modifier{"s" if len(missing) != 1 else ""} for {name} clause.',
                span,
                diagnostics=[
                    f'missing "{field}" modifier'
                    for field in sorted(missing)
                ],
            )

        return cls(span=span, name=name, **kwargs)


    #### CONSTRUCT HELPERS #####################################################

    def _construct_with_rejected[T: tree.Construct](
        self,
        meta: Meta,
        name: Token,
        clause_list: Iterable[tree.Clause|None],
        **extra_args,
    ) -> tuple[T, list[tree.Clause]]:
        """Build a construct and return clauses that it cannot consume.

        Selects the construct class associated with the directive token,
        then assigns compatible clauses to the construct's fields.
        Clauses that do not correspond to a field,
        or whose directive name targets another construct,
        are collected and returned as rejected clauses.

        Repeated clauses are accumulated when the corresponding field is a list.
        Duplicate clauses targeting a non-list field produce a syntax error.
        Required fields annotated as clause types must be supplied.

        Args:
            meta: Parser metadata used to determine the construct's source span.
            name: Token identifying the construct directive.
            clause_list: Candidate clauses to assign to the construct.
                None entries are ignored.
            **extra_args: Additional keyword arguments passed to the constructor.

        Returns:
            A tuple containing the constructed directive
            and a list of clauses that were not consumed by it.

        Raises:
            SyntaxError: If a non-repeatable clause is duplicated
                or a required clause field is missing,
                using the syntax-error mechanism provided by `self.sv`.

        Note:
            Handling of innermost-leaf and outermost-leaf properties is not implemented here.
        """

        cls: type[T] = _DIRECTIVE_TYPES[name.type] # ty:ignore[invalid-assignment] # zuban:ignore[assignment]
        span = self.sv.meta2span(meta)
        type_hints = get_type_hints(cls)

        rejected = []
        kwargs: dict[str, tree.Clause|list[tree.Clause]] = {}
        for clause in clause_list:
            if clause is None:
                continue

            # Reject clauses that do not match a construct field
            # or target another directive through their explicit directive name.
            if (
                not _has_field(clause.id, cls) or
                (clause.directive_name is not None and clause.directive_name.string != cls.id)
            ):
                rejected.append(clause)
                continue

            # A clause with an already populated field may be a repeated clause.
            if clause.id in kwargs:
                current = kwargs[clause.id]

                # If the construct's field is a list, append the new clause
                if isinstance(current, list):
                    current.append(clause) # ty:ignore[invalid-argument-type]

                # Otherwise, means that the construct's field has already been set,
                # so the clause is duplicated. Therefore, raise an error.
                else:
                    raise self.sv.syntax_error(
                        # In the case of "if_" or "for_", remove those underscores
                        f"{clause.id.strip('_')} clause can only be defined once.",
                        clause.span,
                        diagnostics=[("first defined here", current.span)],
                    )

            else:
                # If it wasn't already set, check based on the type
                # whether the clause is repeteable or not.
                hint = type_hints[clause.id]
                original_type = get_origin(hint)
                type_args     = get_args(hint)

                # If the type is something like list[Private] or list[Reduction],
                # it means that this clause is repeatable.
                if original_type is list and type_args and issubclass(type_args[0], tree.Clause):
                    kwargs[clause.id] = [clause]
                else:
                    kwargs[clause.id] = clause

        # Before creating the final object, check if the required fields are set
        missing_required_fields = {
            e
            for e in _required_fields(cls) - kwargs.keys()
            if _is_clause_type(type_hints[e])
        }
        if missing_required_fields:
            raise self.sv.syntax_error(
                f"missing required clause{"s" if len(missing_required_fields) != 1 else ""} for {cls.id} directive.",
                span,
                diagnostics=[
                    f"missing {missing} clause."
                    for missing in sorted(missing_required_fields)
                ],
            )

        return cls(
            span=span,
            name=self._name_from_token(name),
            **kwargs,
            **extra_args
        ), rejected

    def _construct[T: tree.Construct](
        self,
        meta: Meta,
        name: Token,
        clause_list: Iterable[tree.Clause|None],
        **extra_args,
    ) -> T:
        """Build a construct and reject any clauses that it cannot consume.

        Delegates clause assignment and required-field validation to `_construct_with_rejected`.
        If any clauses remain unconsumed,
        raises a syntax error with a diagnostic pointing to each rejected clause.

        Args:
            meta: Parser metadata used to determine the construct's source span.
            name: Token identifying the construct directive.
            clause_list: Candidate clauses to assign to the construct.
                None entries are ignored.
            **extra_args: Additional keyword arguments passed to the constructor.

        Returns:
            The fully constructed directive.

        Raises:
            SyntaxError: If clause assignment fails, a required clause is missing,
                or one or more clauses cannot be used by the construct.
        """
        # Build the construct and retrieve clauses that could not be assigned
        construct, clause_list = self._construct_with_rejected(meta, name, clause_list, **extra_args) # zuban:ignore[var-annotated]

        # Reject any leftover clauses instead of silently discarding them
        if clause_list:
            raise self.sv.syntax_error(
                f"some clauses were not used in this construct.",
                construct.span,
                diagnostics=[
                    (f"{clause.id} clause was not used.", clause.span)
                    for clause in clause_list
                ]
            )
        return construct

    #### TOKENS ################################################################

    @v_args(inline=True)
    def IDENTIFIER(self, token: Token) -> tree.PyName:
        span = self.sv.token2span(token)

        # This ensures that the token is considered an identifier by Python
        if not token.value.isidentifier():
            raise self.sv.syntax_error("invalid characters found in identifier", span)

        return tree.PyName(span=span, string=token.value)

    @v_args(inline=True)
    def INTEGER(self, token: Token) -> tree.PyInt:
        # The int() conversion is safe to do here because the parser guarantees only digits
        # Also, if the base is 0, Python will correctly guess it based on the prefix:
        #     XXX    ==> Base 10
        #     0bXXXX ==> Base 2
        #     0oXXX  ==> Base 8
        #     0xXX   ==> Base 16
        return tree.PyInt(span=self.sv.token2span(token), value=int(token, 0))

    #### PYTHON DEFINITIONS ####################################################

    def py_expr(self, node: Tree) -> tree.PyExpr:
        code, span, source = self._parse_py_code(node.meta)

        if not code.body or len(code.body) != 1 or not isinstance(code.body[0], pyast.Expr):
            raise self.sv.syntax_error("expected expression", span)

        return tree.PyExpr(
            span   = span,
            value  = code.body[0].value,
            source = source,
        )

    def py_stmt(self, node: Tree) -> tree.PyStmt:
        code, span, source = self._parse_py_code(node.meta)

        if not code.body or len(code.body) != 1:
            raise self.sv.syntax_error("expected a single statement", span)

        return tree.PyStmt(
            span   = span,
            value  = code.body[0],
            source = source,
        )

    #### LISTS DEFINITIONS #####################################################

    # var_list: IDENTIFIER ("," IDENTIFIER)*
    def var_list(self, node: Tree) -> list[tree.PyName]:
        return list(cast("list[tree.PyName]", node.children))

    # expr_list: py_expr ("," py_expr)*
    def expr_list(self, node: Tree) -> list[tree.PyExpr]:
        return list(cast("list[tree.PyExpr]", node.children))

    # stmt_list: py_stmt ("," py_stmt)*
    def stmt_list(self, node: Tree) -> list[tree.PyStmt]:
        return list(cast("list[tree.PyStmt]", node.children))

    # Rule to alias from the grammar
    def name(self, node: Tree) -> tree.Name:
        return self._name_from_token(node.children[0])

    @v_args(inline=True)
    def directive_name(self, token: Token) -> tree.DirectiveName:
        return tree.DirectiveName(span=self.sv.token2span(token), string=_DIRECTIVE_TYPES[token.type].id)

    # TODO: may be unused
    def directive_list(self, node: Tree) -> list[tree.DirectiveName]:
        return list(cast("list[tree.DirectiveName]", node.children))


    #### COMMON CLAUSES ########################################################

    # if_clause: IF_CLAUSE "(" [directive_name ":"] py_expr ")"
    def if_clause(self, node: Tree) -> tree.If:
        return self._simple_clause(tree.If, "expr", node)

    # nowait_clause: NOWAIT_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def nowait_clause(self, node: Tree) -> tree.NoWait:
        return self._simple_clause(tree.NoWait, "dont_synchronize", node)

    # collapse_clause: COLLAPSE_CLAUSE "(" [directive_name ":"] py_expr ")"
    def collapse_clause(self, node: Tree) -> tree.Collapse:
        return self._simple_clause(tree.Collapse, "num", node)


    #### data sharing clauses ####

    # default_clause: DEFAULT_CLAUSE "(" data_sharing_attr_name [":" directive_name] ")"
    def default_clause(self, node: Tree) -> tree.Default:
        return self._clause_with_mods(tree.Default, node.meta, node.children[0], node.children[1:])

    # shared_clause: SHARED_CLAUSE "(" [directive_name ":"] var_list ")"
    def shared_clause(self, node: Tree) -> tree.Shared:
        return self._simple_clause(tree.Shared, "targets", node)

    # private_clause: PRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def private_clause(self, node: Tree) -> tree.Private:
        return self._simple_clause(tree.Private, "targets", node)

    # firstprivate_clause: FIRSTPRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def firstprivate_clause(self, node: Tree) -> tree.FirstPrivate:
        return self._simple_clause(tree.FirstPrivate, "targets", node)

    # lastprivate_clause: LASTPRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def lastprivate_clause(self, node: Tree) -> tree.LastPrivate:
        return self._simple_clause(tree.LastPrivate, "targets", node)

    # reduction_clause: REDUCTION_CLAUSE  "(" [directive_name ","] reduction_op ":" var_list ")"
    def reduction_clause(self, node: Tree) -> tree.Reduction:
        return self._clause_with_mods(
            tree.Reduction, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # reduction_op: IDENTIFIER | PLUS | MINUS | MULT | ...
    @v_args(inline=True)
    def reduction_op(self, token: Token|tree.PyName) -> tree.ReductionOp:
        if isinstance(token, tree.PyName):
            return tree.ReductionOp(span=token.span, value=token.string)
        return tree.ReductionOp(span=self.sv.token2span(token), value=str(token))

    # linear_clause: LINEAR_CLAUSE "(" linear_list [":" py_expr] ")"
    # TODO: this does not handle REF/UVAL/VAL properly
    def linear_clause(self, node: Tree) -> tree.Linear:
        return self._clause_with_mods(
            tree.Linear, node.meta, node.children[0], node.children[2:],
            targets=node.children[1]
        )


    #### data copying clauses ####

    # copyin_clause: COPYIN_CLAUSE "(" [directive_name ":"] var_list ")"
    def copyin_clause(self, node: Tree) -> tree.CopyIn:
        return self._simple_clause(tree.CopyIn, "targets", node)

    # copyprivate_clause: COPYPRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def copyprivate_clause(self, node: Tree) -> tree.CopyPrivate:
        return self._simple_clause(tree.CopyPrivate, "targets", node)


    #### master & synch clauses ####

    # depend_clause: DEPEND_CLAUSE "(" [task_dependence_name ":"] expr_list ")"
    def depend_clause(self, node: Tree) -> tree.Depend:
        return self._clause_with_mods(
            tree.Depend, node.meta, node.children[0], node.children[1:-1],
            locator_list=node.children[-1]
        )

    #### data mapping clauses ####

    # map_clause: MAP_CLAUSE "(" [[always_modifier_name ","?] map_type_name ":"] var_list ")"
    def map_clause(self, node: Tree) -> tree.Map:
        return self._clause_with_mods(
            tree.Map, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # defaultmap_clause: DEFAULTMAP_CLAUSE "(" tofrom_modifier_name [":" scalar_modifier_name] ")"
    def defaultmap_clause(self, node: Tree) -> tree.DefaultMap:
        return self._clause_with_mods(tree.DefaultMap, node.meta, node.children[0], node.children[1:])

    #### CONSTRUCT SPECIFIC CLAUSES ############################################

    #### parallel construct clauses ####

    # num_threads_clause: NUM_THREADS_CLAUSE "(" [directive_name  ":"] py_expr ")"
    def num_threads_clause(self, node: Tree) -> tree.NumThreads:
        return self._simple_clause(tree.NumThreads, "nthreads", node)

    # proc_bind_clause: PROC_BIND_CLAUSE "(" [directive_name ":"] (CLOSE | PRIMARY | SPREAD) ")"
    def proc_bind_clause(self, node: Tree) -> tree.ProcBind:
        return self._simple_clause(tree.ProcBind, "affinity_policy_name", node)


    #### for construct clauses ####

    # schedule_type: STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
    @v_args(inline=True)
    def schedule_type(self, token: Token) -> tree.ScheduleType:
        span = self.sv.token2span(token)
        name = tree.Name(span=span, string=str(token))
        return tree.ScheduleType(span=span, kind_name=name)

    # schedule_clause: SCHEDULE_CLAUSE "(" [_schedule_modifier_list ":"] schedule_type ["," py_expr] ")"
    def schedule_clause(self, node: Tree) -> tree.Schedule:
        schedule_pos = -2 if isinstance(node.children[-1], tree.ScheduleType) else -1
        return self._clause_with_mods(
            tree.Schedule, node.meta, node.children[0], node.children[1:schedule_pos],
            type=node.children[schedule_pos],
            chunk=node.children[-1] if schedule_pos == -2 else None,
        )

    # ordered_clause: ORDERED_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def ordered_clause(self, node: Tree) -> tree.OrderedClause:
        return self._simple_clause(tree.OrderedClause, "n", node)


    #### simd construct clauses ####

    # safelen_clause: SAFELEN_CLAUSE "(" [directive_name ":"] py_expr ")"
    def safelen_clause(self, node: Tree) -> tree.SafeLen:
        return self._simple_clause(tree.SafeLen, "length", node)

    # simdlen_clause: SIMDLEN_CLAUSE "(" [directive_name ":"] py_expr ")"
    def simdlen_clause(self, node: Tree) -> tree.Simdlen:
        return self._simple_clause(tree.Simdlen, "length", node)

    # aligned_clause: ALIGNED_CLAUSE "(" var_list [":" _aligned_modifier_list] ")"
    def aligned_clause(self, node: Tree) -> tree.Aligned:
        return self._clause_with_mods(
            tree.Aligned, node.meta, node.children[0], node.children[2:],
            targets=node.children[1]
        )


    #### declare simd construct clauses ####

    # uniform_clause: UNIFORM_CLAUSE "(" [directive_name ":"] var_list ")"
    def uniform_clause(self, node: Tree) -> tree.Uniform:
        return self._simple_clause(tree.Uniform, "targets", node)

    # inbranch_clause: INBRANCH ["(" [directive_name ":"] py_expr ")"]
    def inbranch_clause(self, node: Tree) -> tree.InBranch:
        return self._simple_clause(tree.InBranch, "in_branch", node)

    # notinbranch_clause: NOTINBRANCH ["(" [directive_name ":"] py_expr ")"]
    def notinbranch_clause(self, node: Tree) -> tree.NotInBranch:
        return self._simple_clause(tree.NotInBranch, "not_in_branch", node)


    #### task construct clauses ####

    # final_clause: FINAL_CLAUSE "(" [directive_name ":"] py_expr ")"
    def final_clause(self, node: Tree) -> tree.Final:
        return self._simple_clause(tree.Final, "finalize", node)

    # mergeable_clause: MERGEABLE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def mergeable_clause(self, node: Tree) -> tree.Mergeable:
        return self._simple_clause(tree.Mergeable, "can_merge", node)

    # untied_clause: UNTIED_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def untied_clause(self, node: Tree) -> tree.Untied:
        return self._simple_clause(tree.Untied, "can_change_threads", node)

    # priority_clause: PRIORITY_CLAUSE "(" [directive_name ":"] py_expr ")"
    def priority_clause(self, node: Tree) -> tree.Priority:
        return self._simple_clause(tree.Priority, "value", node)


    #### taskloop construct clauses ####

    # grainsize_clause: GRAINSIZE_CLAUSE "(" [directive_name ":"] py_expr ")"
    def grainsize_clause(self, node: Tree) -> tree.GrainSize:
        return self._simple_clause(tree.GrainSize, "grain_size", node)

    # num_tasks_clause: NUM_TASKS_CLAUSE "(" [_num_tasks_modifier_list ":"] py_expr ")"
    def num_tasks_clause(self, node: Tree) -> tree.NumTasks:
        return self._simple_clause(tree.NumTasks, "num_tasks", node)

    # nogroup_clause: NOGROUP_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def nogroup_clause(self, node: Tree) -> tree.NoGroup:
        return self._simple_clause(tree.NoGroup, "dont_synchronize", node)


    #### target_data construct clauses ####

    # device_clause: DEVICE_CLAUSE "(" [directive_name ":"] py_expr ")"
    def device_clause(self, node: Tree) -> tree.Device:
        return self._simple_clause(tree.Device, "device_description", node)

    # use_device_ptr_clause: USE_DEVICE_PTR_CLAUSE "(" [directive_name ":"] var_list ")"
    def use_device_ptr_clause(self, node: Tree) -> tree.UseDevicePtr:
        return self._simple_clause(tree.UseDevicePtr, "targets", node)


    #### target construct clauses ####

    # is_device_ptr_clause: IS_DEVICE_PTR_CLAUSE "(" [directive_name ":"] var_list ")"
    def is_device_ptr_clause(self, node: Tree) -> tree.IsDevicePtr:
        return self._simple_clause(tree.IsDevicePtr, "targets", node)


    #### target_update construct clauses ####

    # to_clause: TO_CLAUSE "(" [directive_name ":"] var_list ")"
    def to_clause(self, node: Tree) -> tree.To:
        return self._simple_clause(tree.To, "targets", node)

    # from_clause: FROM_CLAUSE "(" [directive_name ":"] var_list ")"
    def from_clause(self, node: Tree) -> tree.From:
        return self._simple_clause(tree.From, "targets", node)


    #### declare target construct clauses ####

    # link_clause: LINK_CLAUSE "(" [directive_name ":"] var_list ")"
    def link_clause(self, node: Tree) -> tree.Link:
        return self._simple_clause(tree.Link, "targets", node)


    #### teams construct clauses ####

    # num_teams_clause: NUM_TEAMS_CLAUSE "(" [directive_name ":"] py_expr ")"
    def num_teams_clause(self, node: Tree) -> tree.NumTeams:
        return self._simple_clause(tree.NumTeams, "upper_bound", node)

    # thread_limit_clause: THREAD_LIMIT_CLAUSE "(" [directive_name ":"] py_expr ")"
    def thread_limit_clause(self, node: Tree) -> tree.ThreadLimit:
        return self._simple_clause(tree.ThreadLimit, "threadlim", node)


    #### distribute construct clauses ####

    # dist_schedule_clause: DIST_SCHEDULE_CLAUSE "(" [directive_name ":"] STATIC ["," py_expr] ")"
    @v_args(inline=True, meta=True)
    def dist_schedule_clause(
        self, meta: Meta, token: Token,
        directive_name: tree.DirectiveName|None,
        static: Token,
        chunk: tree.PyExpr|None,
    ) -> tree.DistSchedule:
        return tree.DistSchedule(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            static = self._name_from_token(static),
            chunk_size = chunk,
            directive_name = directive_name,
        )


    #### critical construct clauses ####

    # hint_clause: HINT_CLAUSE "(" [directive_name ":"] py_expr ")"
    def hint_clause(self, node: Tree) -> tree.Hint:
        return self._simple_clause(tree.Hint, "expr", node)


    #### atomic construct clauses ####

    # seq_cst_clause: SEQ_CST_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def seq_cst_clause(self, node: Tree) -> tree.SeqCst:
        return self._simple_clause(tree.SeqCst, "use_semantics", node)

    # read_clause: READ_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def read_clause(self, node: Tree) -> tree.Read:
        return self._simple_clause(tree.Read, "use_semantics", node)

    # write_clause: WRITE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def write_clause(self, node: Tree) -> tree.Write:
        return self._simple_clause(tree.Write, "use_semantics", node)

    # update_clause: UPDATE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def update_clause(self, node: Tree) -> tree.Update:
        return self._simple_clause(tree.Update, "use_semantics", node)

    # capture_clause: CAPTURE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def capture_clause(self, node: Tree) -> tree.Capture:
        return self._simple_clause(tree.Capture, "use_semantics", node)


    #### ordered construct clauses ####

    # threads_clause: THREADS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def threads_clause(self, node: Tree) -> tree.Threads:
        return self._simple_clause(tree.Threads, "appy_to_threads", node)

    # simd_clause: SIMD_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def simd_clause(self, node: Tree) -> tree.SimdClause:
        return self._simple_clause(tree.SimdClause, "appy_to_simd", node)


    #### declare_reduction construct clauses ####

    # initializer_clause: INITIALIZER_CLAUSE "(" [directive_name ":"] py_stmt ")"
    def initializer_clause(self, node: Tree) -> tree.Initializer:
        return self._simple_clause(tree.Initializer, "initializer_stmt", node)

    # combiner_clause: COMBINER_CLAUSE "(" [directive_name ":"] py_stmt ")"
    def combiner_clause(self, node: Tree) -> tree.Combiner:
        return self._simple_clause(tree.Combiner, "combiner_stmt", node)


    #### SPECIAL CASE CONSTRUCTS ###############################################

    # DECLARE_SIMD_DIRECTIVE ["(" py_expr ")"] _declare_simd_clause_list?
    def declare_simd_directive(self, node: Tree) -> tree.DeclareSimd:
        token, *rest = node.children
        split = next((i for i, c in enumerate(rest) if isinstance(c, tree.Clause)), len(rest))
        fixed, clause_list = rest[:split], rest[split:]

        proc_name = fixed[0] if fixed else None

        return self._construct(node.meta, token, clause_list, proc_name=proc_name) # ty: ignore [invalid-argument-type] # zuban: ignore[arg-type]


    # DECLARE_TARGET_DIRECTIVE _declare_target_clause_list? -> declare_target_directive
    def declare_target_directive(self, node: Tree) -> tree.DeclareTarget:
        return self._construct(
            node.meta,
            cast(Token, node.children[0]),
            cast(Iterable[tree.Clause], node.children[1:])
        )

    # DECLARE_TARGET_DIRECTIVE "(" var_list ")" -> declare_target_directive_list
    def declare_target_directive_list(self, node: Tree) -> tree.DeclareTarget:
        return self._construct(
            node.meta,
            cast(Token, node.children[0]),
            [],
            targets=node.children[1]
        )

    # CRITICAL_DIRECTIVE ["(" IDENTIFIER ")" [","? hint_clause]]
    def critical_directive(self, node: Tree) -> tree.Critical:
        token, *rest = node.children

        if len(rest) == 0:
            name, hint = None, None
        elif len(rest) == 1:
            name, hint = rest[0], None
        else:
            name, hint = rest

        return self._construct(node.meta, token, [hint] if hint is not None else [], critical_name=name) # ty: ignore [invalid-argument-type] # zuban: ignore

    # flush_directive: FLUSH_DIRECTIVE ["(" var_list ")"]
    def flush_directive(self, node: Tree) -> tree.Flush:
        token, *rest = node.children

        clause = next((c for c in rest if isinstance(c, tree.Clause)), None)
        var_list = next((c for c in rest if not isinstance(c, tree.Clause)), None)

        return self._construct(node.meta, token, [clause] if clause is not None else [], targets=var_list) # ty: ignore [invalid-argument-type] # zuban: ignore[arg-type]

    # CANCEL_DIRECTIVE construct_type_clause [","? if_clause]
    def cancel_directive(self, node: Tree) -> tree.Cancel:
        if_clause = node.children[-1] if isinstance(node.children[-1], tree.If) else None
        construct_type = node.children[-1 if if_clause is None else -2]
        return tree.Cancel(
            span=self.sv.meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            construct_type_name=self._name_from_token(construct_type),
            if_=if_clause,
        )

    # CANCELLATION_POINT_DIRECTIVE construct_type_clause
    def cancellation_point_directive(self, node: Tree) -> tree.CancellationPoint:
        return tree.CancellationPoint(
            span=self.sv.meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            construct_type_name=self._name_from_token(node.children[-1]),
        )

    # THREADPRIVATE "(" var_list ")"
    @v_args(inline=True, meta=True)
    def threadprivate_directive(self, meta: Meta, token: Token, targets: list[tree.PyName]) -> tree.ThreadPrivate:
        return self._construct(meta, token, [], targets=targets)

    # DECLARE_REDUCTION_DIRECTIVE "(" reduction_op ":" expr_list ")" _declare_reduction_clause_list
    @v_args(inline=True, meta=True)
    def declare_reduction_directive6(
        self, meta: Meta, token: Token,
        op: tree.ReductionOp, expr_list: list[tree.PyExpr],
        *clause_list: tree.Clause,
    ) -> tree.DeclareReduction:
        return self._construct(meta, token, clause_list, op=op, ann_list=expr_list)

    # DECLARE_REDUCTION_DIRECTIVE "(" reduction_op ":" expr_list ":" py_stmt ")" initializer_clause?
    @v_args(inline=True, meta=True)
    def declare_reduction_directive(
        self, meta: Meta, token: Token,
        op: tree.ReductionOp, type_list: list[tree.PyExpr], py_stmt: tree.PyStmt,
        initializer: tree.Initializer|None=None
    ) -> tree.DeclareReduction:
        span = self.sv.meta2span(meta)
        name = self._name_from_token(token)
        return tree.DeclareReduction(
            span = span,
            name = name,
            op = op,
            ann_list = type_list,
            combiner = tree.Combiner(span=span, name=name, combiner_stmt=py_stmt),
            initializer = initializer,
        )

    ############################################################################

    # combined_directive: combined_directive_name combined_directive_name+ [combined_clause_list]
    def combined_directive(self, node: Tree) -> tree.Directive:
        span = self.sv.meta2span(node.meta)
        if isinstance(node.children[-1], Tree):
            clause_list = cast("list[tree.Clause]", node.children[-1].children)
            directive_list = node.children[:-1]
        else:
            clause_list = []
            directive_list = node.children

        constructs = {}
        for directive_token in directive_list:
            directive_token = cast("Token", directive_token)
            r: tuple[tree.Construct, list[tree.Clause]] = self._construct_with_rejected(node.meta, directive_token, clause_list)
            if r[0].id in constructs:
                raise self.sv.syntax_error(
                    f"{r[0].id} directive appears more than once.",
                    r[0].name.span,
                    diagnostics=[("first defined here.", constructs[r[0].id].name.span)]
                )
            constructs[r[0].id] = r[0]
            clause_list = r[1]

        if clause_list:
            raise self.sv.syntax_error(
                f"some clauses were not used in this combined construct.",
                span,
                diagnostics=[
                    (f"{clause.id} clause was not used.", clause.span)
                    for clause in clause_list
                ]
            )

        return tree.Directive(
            span = span,
            string = self.sv.source_text(self.sv.span), # NOTE: this includes the quotes
            constructs = constructs,
        )

    @v_args(inline=True)
    def start(self, directive: Tree|tree.Directive) -> tree.Directive:
        # If it was already transformed, just return that
        if isinstance(directive, tree.Directive):
            return directive

        if isinstance(directive, tree.Construct):
            span = directive.span
            construct = directive

        if isinstance(directive, Tree):
            # Otherwise, assuming this rule format, process the directive in a general case:
            #   task_directive: TASK_DIRECTIVE _task_clause_list?
            span            = self.sv.meta2span(directive.meta)
            directive_token = cast("Token", directive.children[0])
            clause_list     = cast("list[tree.Clause]", directive.children[1:])
            construct       = self._construct(directive.meta, directive_token, clause_list)

        return tree.Directive(
            span = span,
            string = self.sv.source_text(self.sv.span), # NOTE: this includes the quotes
            constructs = {construct.id: construct},
        )
