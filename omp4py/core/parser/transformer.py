from __future__ import annotations

import ast as pyast
from dataclasses import fields, MISSING
from typing import cast, get_type_hints, get_origin, get_args, Union, Iterable

from . import tree
from .openmp_parser import Transformer, v_args, Token, Tree, Meta
from .source_view import SourceView

__all__ = ["AstTransformer"]

_DIRECTIVE_TYPES = {
    "THREADPRIVATE_DIRECTIVE": tree.ThreadPrivate,
    "DECLARE_REDUCTION_DIRECTIVE": tree.DeclareReduction,
    "DECLARE_INDUCTION_DIRECTIVE": tree.DeclareInduction,
    "SCAN_DIRECTIVE": tree.Scan,
    "DECLARE_MAPPER_DIRECTIVE": tree.DeclareMapper,
    "GROUPPRIVATE_DIRECTIVE": tree.GroupPrivate,
    "ALLOCATE_DIRECTIVE": tree.Allocate,
    "METADIRECTIVE_DIRECTIVE": tree.Metadirective,
    "DECLARE_VARIANT_DIRECTIVE": tree.DeclareVariant,
    "DISPATCH_DIRECTIVE": tree.Dispatch,
    "DECLARE_SIMD_DIRECTIVE": tree.DeclareSimd,
    "DECLARE_TARGET_DIRECTIVE": tree.DeclareTarget,
    "REQUIRES_DIRECTIVE": tree.Requires,
    "ASSUME_DIRECTIVE": tree.Assume,
    "NOTHING_DIRECTIVE": tree.Nothing,
    "ERROR_DIRECTIVE": tree.Error,
    "FUSE_DIRECTIVE": tree.Fuse,
    "INTERCHANGE_DIRECTIVE": tree.Interchange,
    "SPLIT_DIRECTIVE": tree.Split,
    "STRIPE_DIRECTIVE": tree.Stripe,
    "TILE_DIRECTIVE": tree.Tile,
    "UNROLL_DIRECTIVE": tree.Unroll,
    "PARALLEL_DIRECTIVE": tree.Parallel,
    "TEAMS_DIRECTIVE": tree.Teams,
    "SIMD_DIRECTIVE": tree.Simd,
    "MASKED_DIRECTIVE": tree.Masked,
    "SINGLE_DIRECTIVE": tree.Single,
    "SCOPE_DIRECTIVE": tree.Scope,
    "SECTIONS_DIRECTIVE": tree.Sections,
    "SECTION_DIRECTIVE": tree.Section,
    "WORKSHARE_DIRECTIVE": tree.Workshare,
    "WORKDISTRIBUTE_DIRECTIVE": tree.Workdistribute,
    "FOR_DIRECTIVE": tree.For,
    "DISTRIBUTE_DIRECTIVE": tree.Distribute,
    "LOOP_DIRECTIVE": tree.Loop,
    "TASK_DIRECTIVE": tree.Task,
    "TASKLOOP_DIRECTIVE": tree.Taskloop,
    "TASK_ITERATION_DIRECTIVE": tree.TaskIteration,
    "TASKYIELD_DIRECTIVE": tree.Taskyield,
    "TASKGRAPH_DIRECTIVE": tree.Taskgraph,
    "TARGET_DATA_DIRECTIVE": tree.TargetData,
    "TARGET_ENTER_DATA_DIRECTIVE": tree.TargetEnterData,
    "TARGET_EXIT_DATA_DIRECTIVE": tree.TargetExitData,
    "TARGET_DIRECTIVE": tree.Target,
    "TARGET_UPDATE_DIRECTIVE": tree.TargetUpdate,
    "INTEROP_DIRECTIVE": tree.InteropConstruct,
    "CRITICAL_DIRECTIVE": tree.Critical,
    "BARRIER_DIRECTIVE": tree.Barrier,
    "TASKGROUP_DIRECTIVE": tree.Taskgroup,
    "TASKWAIT_DIRECTIVE": tree.Taskwait,
    "ATOMIC_DIRECTIVE": tree.Atomic,
    "FLUSH_DIRECTIVE": tree.Flush,
    "DEPOBJ_DIRECTIVE": tree.Depobj,
    "ORDERED_DIRECTIVE": tree.Ordered,
    "CANCEL_DIRECTIVE": tree.Cancel,
    "CANCELLATION_POINT_DIRECTIVE": tree.CancellationPoint,
}

# TODO: possible lru_cache here
def _has_field(field_name: str, obj: type, ignore: set[str] = {"span", "name"}) -> bool:
    if field_name in ignore:
        return False
    field_names = {f.name for f in fields(obj)}
    return field_name in field_names

# TODO: possible lru_cache here
def _required_fields(cls: type, ignore: set[str] = {"span", "name"}) -> set[str]:
    return {
        f.name
        for f in fields(cls)
        if f.init and f.default is MISSING and f.default_factory is MISSING and f.name not in ignore
    }

def _is_clause_type(hint) -> bool:
    origin = get_origin(hint)
    if origin is list:
        args = get_args(hint)
        return bool(args) and isinstance(args[0], type) and issubclass(args[0], tree.Clause)

    if origin is Union:
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

    # Retrieves the original code and passes it to ast.parse()
    def _parse_py_code(self, meta: Meta) -> tuple[pyast.Module, tree.Span, str]:
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

    # Modifies the Python AST's positions so they are relative to the whole file
    def _apply_span[T: pyast.AST](self, code: T, span: tree.Span, leading_ws: int = 0) -> T:
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

    # Handles clause rules with one optional modifier and a single argument
    # private_clause: PRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    # ==> _simple_clause(tree.Private, "targets", node)
    def _simple_clause[T: tree.Clause](self, cls: type[T], field: str, node: Tree) -> T:
        span = self.sv.meta2span(node.meta)
        name = self._name_from_token(node.children[0])

        directive_name = None
        if len(node.children) == 3 and isinstance(node.children[1], tree.DirectiveName):
            directive_name = node.children[1]
            arg = node.children[2]
        elif len(node.children) == 2:
            arg = node.children[1]
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
        span = self.sv.meta2span(meta)
        assert isinstance(token, Token)
        name = self._name_from_token(token)

        for mod in mod_list:
            if mod is None:
                continue

            field_name: str
            if isinstance(mod, Tree):
                assert _has_field(mod.data, cls), "Incorrect grammar: rule name does not match field"
                assert len(mod.children) == 1 and isinstance(mod.children[0], Token), "Incorrect grammar: modifier rules must be a single token"
                field_name = mod.data
                mod = self._name_from_token(mod.children[0])
            elif isinstance(mod, tree.Modifier):
                field_name = mod.id
            else:
                raise TypeError(f"unexpected modifier kind: {mod!r}")

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
        # TODO: innermost-leaf or outermost-leaf properties are not handled here

        cls: type[T] = _DIRECTIVE_TYPES[name.type] # ty:ignore[invalid-assignment] # zuban:ignore[assignment]
        span = self.sv.meta2span(meta)
        type_hints = get_type_hints(cls)

        rejected = []
        kwargs: dict[str, tree.Clause|list[tree.Clause]] = {}
        for clause in clause_list:
            if clause is None:
                continue

            if (
                not _has_field(clause.id, cls) or
                (clause.directive_name is not None and clause.directive_name.string != name)
            ):
                rejected.append(clause)
                continue

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
        construct, clause_list = self._construct_with_rejected(meta, name, clause_list, **extra_args) # zuban:ignore[var-annotated]
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

    #### COMMON DEFINITIONS ####################################################

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
        return tree.DirectiveName(span=self.sv.token2span(token), string=str(token))

    def directive_list(self, node: Tree) -> list[tree.DirectiveName]:
        return list(cast("list[tree.DirectiveName]", node.children))

    #### MODIFIERS #############################################################

    # reduction_op: IDENTIFIER | PLUS | MINUS | MULT | ...
    @v_args(inline=True)
    def reduction_op(self, token: Token) -> tree.ReductionOp:
        return tree.ReductionOp(span=self.sv.token2span(token), value=str(token))

    # induction_op: IDENTIFIER | PLUS | MULT
    @v_args(inline=True)
    def induction_op(self, token: Token) -> tree.InductionOp:
        return tree.InductionOp(span=self.sv.token2span(token), value=str(token))

    # original_modifier: ORIGINAL "(" (DEFAULT | PRIVATE | SHARED) ")"
    @v_args(inline=True, meta=True)
    def original_modifier(self, meta: Meta, token: Token, sharing_name: Token) -> tree.Original:
        return tree.Original(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            sharing_name=self._name_from_token(sharing_name),
        )

    # iterator_modifier: ITERATOR "(" iterator_specifier ("," iterator_specifier)* ")"
    def iterator_modifier(self, node: Tree) -> tree.Iterator:
        return tree.Iterator(
            span=self.sv.meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            specifiers=cast("list[tree.IteratorSpecifier]", node.children[1:]),
        )

    # iterator_specifier: IDENTIFIER "=" py_expr ":" py_expr [":" py_expr]
    @v_args(inline=True, meta=True)
    def iterator_specifier(
        self, meta: Meta,
        name: tree.PyName,
        begin: tree.PyExpr, end: tree.PyExpr, step: tree.PyExpr|None
    ) -> tree.IteratorSpecifier:
        return tree.IteratorSpecifier(
            span=self.sv.meta2span(meta),
            name=name, begin=begin, end=end, step=step,
        )

    # step_modifier: STEP "(" py_expr ")"
    @v_args(inline=True, meta=True)
    def step_modifier(self, meta: Meta, token: Token, expr: tree.PyExpr) -> tree.Step:
        return tree.Step(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            expr=expr,
        )

    # allocator_modifier: ALLOCATOR "(" py_expr ")"
    @v_args(inline=True, meta=True)
    def allocator_modifier(self, meta: Meta, token: Token, expr: tree.PyExpr) -> tree.AllocatorModifier:
        return tree.AllocatorModifier(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            allocator=expr,
        )

    # align_modifier: ALIGN "(" py_expr ")"
    @v_args(inline=True, meta=True)
    def align_modifier(self, meta: Meta, token: Token, expr: tree.PyExpr) -> tree.AlignModifier:
        return tree.AlignModifier(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            alignment=expr,
        )

    # mapper_modifier: MAPPER "(" IDENTIFIER ")"
    @v_args(inline=True, meta=True)
    def mapper_modifier(self, meta: Meta, token: Token, identifier: tree.PyName) -> tree.Mapper:
        return tree.Mapper(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            identifier=identifier,
        )

    # memspace_modifier: MEMSPACE "(" py_expr ")"
    @v_args(inline=True, meta=True)
    def memspace_modifier(self, meta: Meta, token: Token, handle: tree.PyExpr) -> tree.MemSpace:
        return tree.MemSpace(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            handle=handle,
        )

    # traits_modifier: TRAITS "(" py_expr ")"
    @v_args(inline=True, meta=True)
    def traits_modifier(self, meta: Meta, token: Token, traits: tree.PyExpr) -> tree.Traits:
        return tree.Traits(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            traits=traits,
        )

    # depinfo_modifier: (IN | INOUT| INOUTSET | MUTEXINOUTSET | OUT) "(" var_list ")"
    @v_args(inline=True, meta=True)
    def depinfo_modifier(self, meta: Meta, token: Token, locator_list: list[tree.PyName]) -> tree.DepInfo:
        return tree.DepInfo(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            locator_list=locator_list,
        )

    # loop_modifier: (FUSED | GRID | ...)  ["(" expr_list ")"]
    @v_args(inline=True, meta=True)
    def loop_modifier(self, meta: Meta, token: Token, indices: list[tree.PyExpr]|None) -> tree.LoopModifier:
        return tree.LoopModifier(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            indices=indices or [],
        )

    # FR "(" IDENTIFIER ")" -> fr_selector
    @v_args(inline=True, meta=True)
    def fr_selector(self, meta: Meta, token: Token, identifier: tree.PyName) -> tree.FrSelector:
        return tree.FrSelector(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            identifier = identifier,
        )

    # ATTR "(" expr_list ")" -> attr_selector
    @v_args(inline=True, meta=True)
    def attr_selector(self, meta: Meta, token: Token, expr_list: list[tree.PyExpr]) -> tree.AttrSelector:
        return tree.AttrSelector(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            expr_list = expr_list,
        )

    # prefer_type_modifier: PREFER_TYPE "(" preference_specification ("," preference_specification )* ")"
    # preference_specification: "{" _preference_selector ("," _preference_selector)* "}" | IDENTIFIER
    def prefer_type_modifier(self, node: Tree) -> tree.Prefer:
        return tree.Prefer(
            span=self.sv.meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            spec=[s if isinstance(s, tree.PyName) else list(s.children) for s in node.children[1:]] # ty:ignore[invalid-argument-type] # zuban:ignore[arg-type]
        )

    # append_op: INTEROP "(" _interop_type ("," _interop_type)* ")"
    def append_op(self, node: Tree) -> tree.InteropModifier:
        return tree.InteropModifier(
            span=self.sv.meta2span(node.meta),
            name=self._name_from_token(node.children[0]),
            nkind=cast("list[tree.Name]", node.children[1:])
        )

    # TODO: this uses context_selector as a stmt_list, which is not exactly what the standard required
    # context_selector: stmt_list
    @v_args(inline=True, meta=True)
    def context_selector(self, meta: Meta, stmt_list: list[tree.PyStmt]) -> tree.ContextSelector:
        return tree.ContextSelector(span=self.sv.meta2span(meta), stmt_list=stmt_list)

    # schedule_type: STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
    @v_args(inline=True)
    def schedule_type(self, token: Token) -> tree.ScheduleType:
        span = self.sv.token2span(token)
        name = tree.Name(span=span, string=str(token))
        return tree.ScheduleType(span=span, kind_name=name)

    #### CLAUSES ###############################################################

    # combiner_clause: COMBINER_CLAUSE "(" [directive_name ":"] py_stmt ")"
    def combiner_clause(self, node: Tree) -> tree.Combiner:
        return self._simple_clause(tree.Combiner, "combiner_stmt", node)

    # initializer_clause: INITIALIZER_CLAUSE "(" [directive_name ":"] py_stmt ")"
    def initializer_clause(self, node: Tree) -> tree.Initializer:
        return self._simple_clause(tree.Initializer, "initializer_stmt", node)

    # inductor_clause: INDUCTOR_CLAUSE "(" [directive_name ":"] py_stmt ")"
    def inductor_clause(self, node: Tree) -> tree.Inductor:
        return self._simple_clause(tree.Inductor, "inductor_stmt", node)

    # collector_clause: COLLECTOR_CLAUSE "(" [directive_name ":"] py_expr ")"
    def collector_clause(self, node: Tree) -> tree.Collector:
        return self._simple_clause(tree.Collector, "collector_expr", node)

    # exclusive_clause: EXCLUSIVE_CLAUSE "(" [directive_name ":"] var_list ")"
    def exclusive_clause(self, node: Tree) -> tree.Exclusive:
        return self._simple_clause(tree.Exclusive, "targets", node)

    # inclusive_clause: INCLUSIVE_CLAUSE "(" [directive_name ":"] var_list ")"
    def inclusive_clause(self, node: Tree) -> tree.Inclusive:
        return self._simple_clause(tree.Inclusive, "targets", node)

    # init_complete_clause: INIT_COMPLETE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def init_complete_clause(self, node: Tree) -> tree.InitComplete:
        return self._simple_clause(tree.InitComplete, "create_init_phase", node)

    # device_type_clause: DEVICE_TYPE_CLAUSE "(" [directive_name ":"] device_type_kind ")"
    def device_type_clause(self, node: Tree) -> tree.DeviceType:
        return self._simple_clause(tree.DeviceType, "ndevice_type_description", node)

    # align_clause: ALIGN_CLAUSE "(" [directive_name ":"] py_expr ")"
    def align_clause(self, node: Tree) -> tree.Align:
        return self._simple_clause(tree.Align, "alignment", node)

    # allocator_clause: ALLOCATOR_CLAUSE "(" [directive_name ":"] py_expr ")"
    def allocator_clause(self, node: Tree) -> tree.Allocator:
        return self._simple_clause(tree.Allocator, "allocator", node)

    # when_clause: WHEN_CLAUSE "(" _when_modifier_list ":" start ")"
    def when_clause(self, node: Tree) -> tree.When:
        return self._clause_with_mods(
            tree.When, node.meta, node.children[0], node.children[1:-1],
            directive=node.children[-1]
        )

    # otherwise_clause: OTHERWISE_CLAUSE ["(" [directive_name ":"] start ")"]
    def otherwise_clause(self, node: Tree) -> tree.Otherwise:
        return self._simple_clause(tree.Otherwise, "directive", node)

    # adjust_args_clause: ADJUST_ARGS_CLAUSE "(" _adjust_args_modifier_list ":" var_list ")"
    def adjust_args_clause(self, node: Tree) -> tree.AdjustArgs:
        return self._clause_with_mods(
            tree.AdjustArgs, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # append_args_clause: APPEND_ARGS_CLAUSE "(" [directive_name ":"] append_args_arg ")"
    def append_args_clause(self, node: Tree) -> tree.AppendArgs:
        return self._simple_clause(tree.AppendArgs, "append_op", node)

    # match_clause: MATCH_CLAUSE "(" [directive_name ":"] context_selector ")"
    def match_clause(self, node: Tree) -> tree.Match:
        return self._simple_clause(tree.Match, "context_selector", node)

    # interop_clause: INTEROP_CLAUSE "(" [directive_name ":"] var_list ")"
    def interop_clause(self, node: Tree) -> tree.InteropClause:
        return self._simple_clause(tree.InteropClause, "targets", node)

    # is_device_ptr_clause: IS_DEVICE_PTR_CLAUSE "(" [directive_name ":"] var_list ")"
    def is_device_ptr_clause(self, node: Tree) -> tree.IsDevicePtr:
        return self._simple_clause(tree.IsDevicePtr, "targets", node)

    # has_device_addr_clause: HAS_DEVICE_ADDR_CLAUSE "(" [directive_name ":"] var_list ")"
    def has_device_addr_clause(self, node: Tree) -> tree.HasDeviceAddr:
        return self._simple_clause(tree.HasDeviceAddr, "targets", node)

    # nocontext_clause: NOCONTEXT_CLAUSE "(" [directive_name ":"] py_expr ")"
    def nocontext_clause(self, node: Tree) -> tree.NoContext:
        return self._simple_clause(tree.NoContext, "dont_update_context", node)

    # novariants_clause: NOVARIANTS_CLAUSE "(" [directive_name ":"] py_expr ")"
    def novariants_clause(self, node: Tree) -> tree.NoVariants:
        return self._simple_clause(tree.NoVariants, "dont_use_variant", node)

    # aligned_clause: ALIGNED_CLAUSE "(" var_list [":" _aligned_modifier_list] ")"
    def aligned_clause(self, node: Tree) -> tree.Aligned:
        return self._clause_with_mods(
            tree.Aligned, node.meta, node.children[0], node.children[2:],
            targets=node.children[1]
        )

    # linear_clause: LINEAR_CLAUSE "(" var_list [":" _linear_modifier_list] ")"
    def linear_clause(self, node: Tree) -> tree.Linear:
        return self._clause_with_mods(
            tree.Linear, node.meta, node.children[0], node.children[2:],
            targets=node.children[1]
        )

    # simdlen_clause: SIMDLEN_CLAUSE "(" [directive_name ":"] py_expr ")"
    def simdlen_clause(self, node: Tree) -> tree.Simdlen:
        return self._simple_clause(tree.Simdlen, "length", node)

    # uniform_clause: UNIFORM_CLAUSE "(" [directive_name ":"] var_list ")"
    def uniform_clause(self, node: Tree) -> tree.Uniform:
        return self._simple_clause(tree.Uniform, "targets", node)

    # inbranch_clause: INBRANCH ["(" [directive_name ":"] py_expr ")"]
    def inbranch_clause(self, node: Tree) -> tree.InBranch:
        return self._simple_clause(tree.InBranch, "in_branch", node)

    # notinbranch_clause: NOTINBRANCH ["(" [directive_name ":"] py_expr ")"]
    def notinbranch_clause(self, node: Tree) -> tree.NotInBranch:
        return self._simple_clause(tree.NotInBranch, "not_in_branch", node)

    # enter_clause: ENTER_CLAUSE "(" [_enter_modifier_list ":"] var_list ")"
    def enter_clause(self, node: Tree) -> tree.Enter:
        return self._clause_with_mods(
            tree.Enter, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # indirect_clause: INDIRECT_CLAUSE ["(" [directive_name ":"] py_type ")"]
    def indirect_clause(self, node: Tree) -> tree.Indirect:
        return self._simple_clause(tree.Indirect, "invoked_by_fptr", node)

    # link_clause: LINK_CLAUSE "(" [directive_name ":"] var_list ")"
    def link_clause(self, node: Tree) -> tree.Link:
        return self._simple_clause(tree.Link, "targets", node)

    # local_clause: LOCAL_CLAUSE "(" [directive_name ":"] var_list ")"
    def local_clause(self, node: Tree) -> tree.Local:
        return self._simple_clause(tree.Local, "targets", node)

    # atomic_default_mem_order_clause: ATOMIC_DEFAULT_MEM_ORDER_CLAUSE "(" [directive_name ":"] (ACQ_REL | ACQUIRE | RELAXED | SEQ_CST) ")"
    def atomic_default_mem_order_clause(self, node: Tree) -> tree.AtomicDefaultMemOrder:
        return self._simple_clause(tree.AtomicDefaultMemOrder, "nmemory_order", node)

    # dynamic_allocators_clause: DYNAMIC_ALLOCATORS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def dynamic_allocators_clause(self, node: Tree) -> tree.DynamicAllocators:
        return self._simple_clause(tree.DynamicAllocators, "required", node)

    # reverse_offload_clause: REVERSE_OFFLOAD_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def reverse_offload_clause(self, node: Tree) -> tree.ReverseOffload:
        return self._simple_clause(tree.ReverseOffload, "required", node)

    # unified_address_clause: UNIFIED_ADDRESS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def unified_address_clause(self, node: Tree) -> tree.UnifiedAddress:
        return self._simple_clause(tree.UnifiedAddress, "required", node)

    # unified_shared_memory_clause: UNIFIED_SHARED_MEMORY_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def unified_shared_memory_clause(self, node: Tree) -> tree.UnifiedSharedMemory:
        return self._simple_clause(tree.UnifiedSharedMemory, "required", node)

    # self_maps_clause: SELF_MAPS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def self_maps_clause(self, node: Tree) -> tree.SelfMaps:
        return self._simple_clause(tree.SelfMaps, "required", node)

    # device_safesync_clause: DEVICE_SAFESYNC_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def device_safesync_clause(self, node: Tree) -> tree.DeviceSafesync:
        return self._simple_clause(tree.DeviceSafesync, "required", node)

    # absent_clause: ABSENT_CLAUSE "(" [directive_name ":"] directive_list ")"
    def absent_clause(self, node: Tree) -> tree.Absent:
        return self._simple_clause(tree.Absent, "directive_names", node)

    # contains_clause: CONTAINS_CLAUSE "(" [directive_name ":"] directive_list ")"
    def contains_clause(self, node: Tree) -> tree.Contains:
        return self._simple_clause(tree.Contains, "directive_names", node)

    # holds_clause: HOLDS_CLAUSE "(" [directive_name ":"] py_expr ")"
    def holds_clause(self, node: Tree) -> tree.Holds:
        return self._simple_clause(tree.Holds, "hold_expr", node)

    # no_openmp_clause: NO_OPENMP_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def no_openmp_clause(self, node: Tree) -> tree.NoOpenmp:
        return self._simple_clause(tree.NoOpenmp, "can_assume", node)

    # no_openmp_constructs_clause: NO_OPENMP_CONSTRUCTS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def no_openmp_constructs_clause(self, node: Tree) -> tree.NoOpenmpConstructs:
        return self._simple_clause(tree.NoOpenmpConstructs, "can_assume", node)

    # no_openmp_routines_clause: NO_OPENMP_ROUTINES_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def no_openmp_routines_clause(self, node: Tree) -> tree.NoOpenmpRoutines:
        return self._simple_clause(tree.NoOpenmpRoutines, "can_assume", node)

    # no_parallelism_clause: NO_PARALLELISM_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def no_parallelism_clause(self, node: Tree) -> tree.NoParallelism:
        return self._simple_clause(tree.NoParallelism, "can_assume", node)

    # at_clause: AT_CLAUSE "(" [directive_name ":"] (COMPILATION | EXECUTION) ")"
    def at_clause(self, node: Tree) -> tree.At:
        return self._simple_clause(tree.At, "naction_time", node)

    # message_clause: MESSAGE_CLAUSE "(" [directive_name ":"] py_expr ")"
    def message_clause(self, node: Tree) -> tree.Message:
        return self._simple_clause(tree.Message, "msg_string", node)

    # severity_clause: SEVERITY_CLAUSE "(" [directive_name ":"] (FATAL | WARNING) ")"
    def severity_clause(self, node: Tree) -> tree.Severity:
        return self._simple_clause(tree.Severity, "nseverity_level", node)

    # looprange_clause: LOOPRANGE_CLAUSE "(" [directive_name ":"] py_expr "," py_expr ")"
    @v_args(inline=True, meta=True)
    def looprange_clause(
        self, meta: Meta, token: Token,
        directive_name: tree.DirectiveName|None, first: tree.PyExpr, count: tree.PyExpr,
    ) -> tree.LoopRange:
        return tree.LoopRange(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            first = first,
            count = count,
            directive_name = directive_name,
        )

    # permutation_clause: PERMUTATION_CLAUSE "(" [directive_name ":"] expr_list ")"
    def permutation_clause(self, node: Tree) -> tree.Permutation:
        return self._simple_clause(tree.Permutation, "permutation_list", node)

    # counts_clause: COUNTS_CLAUSE "(" [directive_name ":"] expr_list ")"
    def counts_clause(self, node: Tree) -> tree.Counts:
        return self._simple_clause(tree.Counts, "count_list", node)

    # sizes_clause: SIZES_CLAUSE "(" [directive_name ":"] expr_list ")"
    def sizes_clause(self, node: Tree) -> tree.Sizes:
        return self._simple_clause(tree.Sizes, "size_list", node)

    # full_clause: FULL_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def full_clause(self, node: Tree) -> tree.Full:
        return self._simple_clause(tree.Full, "fully_unroll", node)

    # partial_clause: PARTIAL_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def partial_clause(self, node: Tree) -> tree.Partial:
        return self._simple_clause(tree.Partial, "unroll_factor", node)

    # copyin_clause: COPYIN_CLAUSE "(" [directive_name ":"] var_list ")"
    def copyin_clause(self, node: Tree) -> tree.CopyIn:
        return self._simple_clause(tree.CopyIn, "targets", node)

    # num_threads_clause: NUM_THREADS_CLAUSE "(" [_num_threads_modifier_list ":"] expr_list ")"
    def num_threads_clause(self, node: Tree) -> tree.NumThreads:
        return self._clause_with_mods(
            tree.NumThreads, node.meta, node.children[0], node.children[1:-1],
            upper_bound=node.children[-1]
        )

    # proc_bind_clause: PROC_BIND_CLAUSE "(" [directive_name ":"] (CLOSE | PRIMARY | SPREAD) ")"
    def proc_bind_clause(self, node: Tree) -> tree.ProcBind:
        return self._simple_clause(tree.ProcBind, "naffinity_policy", node)

    # safesync_clause: SAFESYNC_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def safesync_clause(self, node: Tree) -> tree.SafeSync:
        return self._simple_clause(tree.SafeSync, "width", node)

    # num_teams_clause: NUM_TEAMS_CLAUSE "(" [_num_teams_modifier_list ":"] py_expr ")"
    def num_teams_clause(self, node: Tree) -> tree.NumTeams:
        return self._clause_with_mods(
            tree.NumTeams, node.meta, node.children[0], node.children[1:-1],
            upper_bound=node.children[-1]
        )

    # thread_limit_clause: THREAD_LIMIT_CLAUSE "(" [directive_name ":"] py_expr ")"
    def thread_limit_clause(self, node: Tree) -> tree.ThreadLimit:
        return self._simple_clause(tree.ThreadLimit, "threadlim", node)

    # nontemporal_clause: NONTEMPORAL_CLAUSE "(" [directive_name ":"] var_list ")"
    def nontemporal_clause(self, node: Tree) -> tree.NonTemporal:
        return self._simple_clause(tree.NonTemporal, "targets", node)

    # order_clause: ORDER_CLAUSE "(" [_order_modifier_list ":"] CONCURRENT ")"
    def order_clause(self, node: Tree) -> tree.Order:
        return self._clause_with_mods(
            tree.Order, node.meta, node.children[0], node.children[1:-1],
            ordering=self._name_from_token(node.children[-1])
        )

    # safelen_clause: SAFELEN_CLAUSE "(" [directive_name ":"] py_expr ")"
    def safelen_clause(self, node: Tree) -> tree.SafeLen:
        return self._simple_clause(tree.SafeLen, "length", node)

    # filter_clause: FILTER_CLAUSE "(" [directive_name ":"] py_expr ")"
    def filter_clause(self, node: Tree) -> tree.Filter:
        return self._simple_clause(tree.Filter, "thread_num", node)

    # copyprivate_clause: COPYPRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def copyprivate_clause(self, node: Tree) -> tree.CopyPrivate:
        return self._simple_clause(tree.CopyPrivate, "targets", node)

    # ordered_clause: ORDERED_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def ordered_clause(self, node: Tree) -> tree.OrderedClause:
        return self._simple_clause(tree.OrderedClause, "n", node)

    # schedule_clause: SCHEDULE_CLAUSE "(" [_schedule_modifier_list ":"] schedule_type ["," py_expr] ")"
    def schedule_clause(self, node: Tree) -> tree.Schedule:
        schedule_pos = -2 if isinstance(node.children[-1], tree.ScheduleType) else -1
        return self._clause_with_mods(
            tree.Schedule, node.meta, node.children[0], node.children[1:schedule_pos],
            type=node.children[schedule_pos],
            chunk=node.children[-1] if schedule_pos == -2 else None,
        )

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

    # bind_clause: BIND_CLAUSE "(" [directive_name ":"] _bind_clause_arg ")"
    def bind_clause(self, node: Tree) -> tree.Bind:
        return self._simple_clause(tree.Bind, "nbinding", node)

    # grainsize_clause: GRAINSIZE_CLAUSE "(" [_grainsize_modifier_list ":"] py_expr ")"
    def grainsize_clause(self, node: Tree) -> tree.GrainSize:
        return self._clause_with_mods(
            tree.GrainSize, node.meta, node.children[0], node.children[1:-1],
            grain_size=node.children[-1]
        )

    # num_tasks_clause: NUM_TASKS_CLAUSE "(" [_num_tasks_modifier_list ":"] py_expr ")"
    def num_tasks_clause(self, node: Tree) -> tree.NumTasks:
        return self._clause_with_mods(
            tree.NumTasks, node.meta, node.children[0], node.children[1:-1],
            grain_size=node.children[-1]
        )

    # graph_id_clause: GRAPH_ID_CLAUSE "(" [directive_name ":"] py_expr ")"
    def graph_id_clause(self, node: Tree) -> tree.GraphId:
        return self._simple_clause(tree.GraphId, "graph_id_value", node)

    # graph_reset_clause: GRAPH_RESET_CLAUSE "(" [directive_name ":"] py_expr ")"
    def graph_reset_clause(self, node: Tree) -> tree.GraphReset:
        return self._simple_clause(tree.GraphReset, "expr", node)

    # use_device_ptr_clause: USE_DEVICE_PTR_CLAUSE "(" [directive_name ":"] var_list ")"
    def use_device_ptr_clause(self, node: Tree) -> tree.UseDevicePtr:
        return self._simple_clause(tree.UseDevicePtr, "targets", node)

    # use_device_addr_clause: USE_DEVICE_ADDR_CLAUSE "(" [directive_name ":"] var_list ")"
    def use_device_addr_clause(self, node: Tree) -> tree.UseDeviceAddr:
        return self._simple_clause(tree.UseDeviceAddr, "targets", node)

    # defaultmap_clause: DEFAULTMAP_CLAUSE "(" _defaultmap_arg [":" _defaultmap_modifier_list] ")"
    def defaultmap_clause(self, node: Tree) -> tree.DefaultMap:
        return self._clause_with_mods(
            tree.DefaultMap, node.meta, node.children[0], node.children[2:],
            implicit_behavior_name=self._name_from_token(node.children[1])
        )

    # uses_allocators_clause: USES_ALLOCATORS_CLAUSE "(" [_uses_allocator_modifier_list ":"] py_expr ")"
    def uses_allocators_clause(self, node: Tree) -> tree.UsesAllocators:
        return self._clause_with_mods(
            tree.UsesAllocators, node.meta, node.children[0], node.children[1:-1],
            grain_size=node.children[-1]
        )

    # to_clause: TO_CLAUSE "(" [_to_modifier_list ":"] var_list ")"
    def to_clause(self, node: Tree) -> tree.To:
        return self._clause_with_mods(
            tree.To, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # from_clause: FROM_CLAUSE "(" [_from_modifier_list ":"] var_list ")"
    def from_clause(self, node: Tree) -> tree.From:
        return self._clause_with_mods(
            tree.From, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # destroy_clause: DESTROY_CLAUSE "(" [directive_name ":"] IDENTIFIER ")"
    def destroy_clause(self, node: Tree) -> tree.Destroy:
        return self._simple_clause(tree.Destroy, "destroy_var", node)

    # init_clause: INIT_CLAUSE "(" [_init_modifier_list ":"] IDENTIFIER ")"
    def init_clause(self, node: Tree) -> tree.Init:
        span = self.sv.meta2span(node.meta)
        kwargs: dict[str, tree.Construct|list[tree.Name]] = {}
        for mod in node.children[1:-1]:
            if mod is None:
                continue

            # interop_type_modifier_name
            if isinstance(mod, tree.Name):
                if "interop_type_modifier_name" not in kwargs:
                    kwargs["interop_type_modifier_name"] = [mod]
                else:
                    assert isinstance(kwargs["interop_type_modifier_name"], list)
                    kwargs["interop_type_modifier_name"].append(mod) # ty:ignore[invalid-argument-type]

            # prefer_type_modifier
            if not isinstance(mod, tree.Prefer):
                continue

            # depinfo_modifier
            if not isinstance(mod, tree.DepInfo):
                continue

            # directive_name_modifier
            if not isinstance(mod, tree.DirectiveName):
                continue

            if mod.id in kwargs:
                raise self.sv.syntax_error(
                    f'"{mod.id}" modifier is already defined for init clause.', span,
                    diagnostics=[("first defined here", kwargs[mod.id].span)] # ty:ignore[unresolved-attribute] # zuban:ignore[union-attr]
                )
            kwargs[mod.id] = mod # ty:ignore[invalid-assignment] # zuban:ignore[assignment]

        return tree.Init(
            span = span,
            name = self._name_from_token(node.children[0]),
            init_var = cast("tree.PyName", node.children[-1]),
            **kwargs # ty:ignore[invalid-argument-type] # zuban:ignore[arg-type]
        )

    # use_clause: USE_CLAUSE "(" [directive_name ":"] IDENTIFIER ")"
    def use_clause(self, node: Tree) -> tree.Use:
        return self._simple_clause(tree.Use, "interop_var", node)

    # hint_clause: HINT_CLAUSE "(" [directive_name ":"] py_expr ")"
    def hint_clause(self, node: Tree) -> tree.Hint:
        return self._simple_clause(tree.Hint, "expr", node)

    # task_reduction_clause: TASK_REDUCTION_CLAUSE "(" [directive_name ","] reduction_op ":" var_list ")"
    @v_args(inline=True, meta=True)
    def task_reduction_clause(
        self, meta: Meta, token: Token,
        directive_name: tree.DirectiveName|None,
        reduction_op: tree.ReductionOp,
        var_list: list[tree.PyName],
    ) -> tree.TaskReduction:
        return tree.TaskReduction(
            span=self.sv.meta2span(meta),
            name=self._name_from_token(token),
            op = reduction_op,
            targets = var_list,
            directive_name = directive_name,
        )

    # memscope_clause: MEMSCOPE_CLAUSE "(" [directive_name ":"] (ALL | CGROUP | DEVICE) ")"
    def memscope_clause(self, node: Tree) -> tree.MemScope:
        return self._simple_clause(tree.MemScope, "nscope", node)

    # read_clause: READ_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def read_clause(self, node: Tree) -> tree.Read:
        return self._simple_clause(tree.Read, "", node)

    # atomic_update_clause: UPDATE_CLAUSE ["(" [directive_name ":"] py_expr ")"] // innermost-leaf, unique
    def atomic_update_clause(self, node: Tree) -> tree.Update:
        return self._simple_clause(tree.Update, "use_semantics", node)

    # write_clause: WRITE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def write_clause(self, node: Tree) -> tree.Write:
        return self._simple_clause(tree.Write, "use_semantics", node)

    # capture_clause: CAPTURE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def capture_clause(self, node: Tree) -> tree.Capture:
        return self._simple_clause(tree.Capture, "use_semantics", node)

    # compare_clause: COMPARE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def compare_clause(self, node: Tree) -> tree.Compare:
        return self._simple_clause(tree.Compare, "use_semantics", node)

    # fail_clause: FAIL_CLAUSE "(" [directive_name ":"] (ACQUIRE | RELAXED | SEQ_CST) ")"
    def fail_clause(self, node: Tree) -> tree.Fail:
        return self._simple_clause(tree.Fail, "nmem_order", node)

    # weak_clause: WEAK_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def weak_clause(self, node: Tree) -> tree.Weak:
        return self._simple_clause(tree.Weak, "use_semantics", node)

    # acq_rel_clause: ACQ_REL_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def acq_rel_clause(self, node: Tree) -> tree.AcqRel:
        return self._simple_clause(tree.AcqRel, "use_semantics", node)

    # acquire_clause: ACQUIRE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def acquire_clause(self, node: Tree) -> tree.Acquire:
        return self._simple_clause(tree.Acquire, "use_semantics", node)

    # relaxed_clause: RELAXED_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def relaxed_clause(self, node: Tree) -> tree.Relaxed:
        return self._simple_clause(tree.Relaxed, "use_semantics", node)

    # release_clause: RELEASE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def release_clause(self, node: Tree) -> tree.Release:
        return self._simple_clause(tree.Release, "use_semantics", node)

    # seq_cst_clause: SEQ_CST_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def seq_cst_clause(self, node: Tree) -> tree.SeqCst:
        return self._simple_clause(tree.SeqCst, "use_semantics", node)

    # depobj_update_clause: UPDATE_CLAUSE "(" [_depobj_update_modifier_list ":"] IDENTIFIER ")"
    def depobj_update_clause(self, node: Tree) -> tree.DepobjUpdate:
        return self._clause_with_mods(
            tree.DepobjUpdate, node.meta, node.children[0], node.children[1:-1],
            update_var=node.children[-1]
        )
    # doacross_clause: DOACROSS_CLAUSE "(" _doacross_modifier_list ":" _iterator_specifier ")"
    def doacross_clause(self, node: Tree) -> tree.DoAcross:
        return self._clause_with_mods(
            tree.DoAcross, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # threads_clause: THREADS_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def threads_clause(self, node: Tree) -> tree.Threads:
        return self._simple_clause(tree.Threads, "appy_to_threads", node)

    # simd_clause: SIMD_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def simd_clause(self, node: Tree) -> tree.SimdClause:
        return self._simple_clause(tree.SimdClause, "appy_to_simd", node)

    #### COMMON CLAUSES ########################################################

    # apply_clause: APPLY_CLAUSE "(" [_apply_modifier_list ":"] apply_clause_arg ")"
    def apply_clause(self, node: Tree) -> tree.Apply:
        return self._clause_with_mods(
            tree.Apply, node.meta, node.children[0], node.children[1:-1],
            directives=[self._name_from_token(t) for t in node.children[-1].children]
        )

    # depend_clause: DEPEND_CLAUSE "(" [_depend_modifier_list ":"] expr_list ")"
    def depend_clause(self, node: Tree) -> tree.Depend:
        return self._clause_with_mods(
            tree.Depend, node.meta, node.children[0], node.children[1:-1],
            locator_list=node.children[-1]
        )

    # device_clause: DEVICE_CLAUSE "(" [_device_modifier_list ":"] py_expr ")"
    def device_clause(self, node: Tree) -> tree.Device:
        return self._clause_with_mods(
            tree.Device, node.meta, node.children[0], node.children[1:-1],
            directive=node.children[-1]
        )

    # default_clause: DEFAULT_CLAUSE "(" (NONE | SHARED | FIRSTPRIVATE | PRIVATE) [":" _default_modifier] ")"
    def default_clause(self, node: Tree) -> tree.When:
        return self._clause_with_mods(
            tree.When, node.meta, node.children[0], node.children[2:],
            data_sharing_attr_name=self._name_from_token(node.children[1]),
        )

    # private_clause: PRIVATE_CLAUSE "(" [directive_name ":"] var_list ")"
    def private_clause(self, node: Tree) -> tree.Private:
        return self._simple_clause(tree.Private, "targets", node)

    # if_clause: IF_CLAUSE "(" [directive_name ":"] py_expr ")"
    def if_clause(self, node: Tree) -> tree.If:
        return self._simple_clause(tree.If, "expr", node)

    # firstprivate_clause: FIRSTPRIVATE_CLAUSE "(" [_firstprivate_modifier ":"] var_list ")"
    def firstprivate_clause(self, node: Tree) -> tree.FirstPrivate:
        return self._clause_with_mods(
            tree.FirstPrivate, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # reduction_clause: REDUCTION_CLAUSE  "(" [_reduction_modifier_list ","] reduction_op ":" var_list ")"
    def reduction_clause(self, node: Tree) -> tree.Reduction:
        return self._clause_with_mods(
            tree.Reduction, node.meta, node.children[0], node.children[1:-2],
            targets=node.children[-1],
            op=node.children[-2],
        )

    # induction_clause: INDUCTION_CLAUSE "(" _induction_modifier_list "," induction_op ":" var_list ")"
    def induction_clause(self, node: Tree) -> tree.Induction:
        return self._clause_with_mods(
            tree.Induction, node.meta, node.children[0], node.children[1:-2],
            targets=node.children[-1],
            op=node.children[-2],
        )

    # shared_clause: SHARED_CLAUSE "(" [directive_name ":"] var_list ")"
    def shared_clause(self, node: Tree) -> tree.Shared:
        return self._simple_clause(tree.Shared, "targets", node)

    # collapse_clause: COLLAPSE_CLAUSE "(" [directive_name ":"] py_expr ")"
    def collapse_clause(self, node: Tree) -> tree.Collapse:
        return self._simple_clause(tree.Collapse, "num", node)

    # lastprivate_clause: LASTPRIVATE_CLAUSE "(" [_lastprivate_modifier_list ":"] var_list ")"
    def lastprivate_clause(self, node: Tree) -> tree.LastPrivate:
        return self._clause_with_mods(
            tree.LastPrivate, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # allocate_clause: ALLOCATE_CLAUSE "(" [_allocate_modifier_list ":"] var_list ")"
    def allocate_clause(self, node: Tree) -> tree.AllocateClause:
        return self._clause_with_mods(
            tree.AllocateClause, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # nowait_clause: NOWAIT_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def nowait_clause(self, node: Tree) -> tree.NoWait:
        return self._simple_clause(tree.NoWait, "dont_synchronize", node)

    # final_clause: FINAL_CLAUSE "(" [directive_name ":"] py_expr ")"
    def final_clause(self, node: Tree) -> tree.Final:
        return self._simple_clause(tree.Final, "finalize", node)

    # mergeable_clause: MERGEABLE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def mergeable_clause(self, node: Tree) -> tree.Mergeable:
        return self._simple_clause(tree.Mergeable, "can_merge", node)

    # untied_clause: UNTIED_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def untied_clause(self, node: Tree) -> tree.Untied:
        return self._simple_clause(tree.Untied, "can_change_threads", node)

    # affinity_clause: AFFINITY_CLAUSE "(" [_affinity_modifier_list ":"] var_list ")"
    def affinity_clause(self, node: Tree) -> tree.When:
        return self._clause_with_mods(
            tree.When, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    # detach_clause: DETACH_CLAUSE "(" [directive_name ":"] IDENTIFIER ")"
    def detach_clause(self, node: Tree) -> tree.Detach:
        return self._simple_clause(tree.Detach, "event_handle", node)

    # in_reduction_clause: IN_REDUCTION_CLAUSE "(" [directive_name ","] reduction_op ":" var_list ")"
    def in_reduction_clause(self, node: Tree) -> tree.InReduction:
        return self._clause_with_mods(
            tree.InReduction, node.meta, node.children[0], node.children[1:-2],
            targets=node.children[-1],
            op=node.children[-2],
        )

    # priority_clause: PRIORITY_CLAUSE "(" [directive_name ":"] py_expr ")"
    def priority_clause(self, node: Tree) -> tree.Priority:
        return self._simple_clause(tree.Priority, "value", node)

    # replayable_clause: REPLAYABLE_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def replayable_clause(self, node: Tree) -> tree.Replayable:
        return self._simple_clause(tree.Replayable, "expr", node)

    # threadset_clause: THREADSET_CLAUSE "(" [directive_name ":"] (OMP_TEAM | OMP_POOL) ")"
    def threadset_clause(self, node: Tree) -> tree.ThreadSet:
        return self._simple_clause(tree.ThreadSet, "nset", node)

    # transparent_clause: TRANSPARENT_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def transparent_clause(self, node: Tree) -> tree.Transparent:
        return self._simple_clause(tree.Transparent, "impex_type", node)

    # nogroup_clause: NOGROUP_CLAUSE ["(" [directive_name ":"] py_expr ")"]
    def nogroup_clause(self, node: Tree) -> tree.NoGroup:
        return self._simple_clause(tree.NoGroup, "dont_synchronize", node)

    # map_clause: MAP_CLAUSE "(" [[_map_modifier_list ","] map_type_name ":"] var_list ")"
    def map_clause(self, node: Tree) -> tree.Map:
        return self._clause_with_mods(
            tree.Map, node.meta, node.children[0], node.children[1:-1],
            targets=node.children[-1]
        )

    #### SPECIAL CASE CONSTRUCTS ###############################################

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
        initializer: tree.Initializer|None
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

    # DECLARE_INDUCTION_DIRECTIVE "(" induction_op ":" expr_list ")" _declare_induction_clause_list
    @v_args(inline=True, meta=True)
    def declare_induction_directive(
        self, meta: Meta, token: Token,
        op: tree.InductionOp, expr_list: list[tree.PyExpr],
        *clause_list: tree.Clause,
    ) -> tree.DeclareInduction:
        return self._construct(meta, token, clause_list, op=op, ann_list=expr_list)

    # DECLARE_MAPPER_DIRECTIVE "(" [IDENTIFIER ":"] IDENTIFIER ":" py_type ")" map_clause+
    @v_args(inline=True, meta=True)
    def declare_mapper_directive(
        self, meta: Meta, token: Token,
        mapper: tree.PyName|None, var: tree.PyName, type: tree.PyExpr,
        *clause_list: tree.Clause,
    ) -> tree.DeclareMapper:
        return self._construct(meta, token, clause_list, mapper_identifier=mapper, var=var, type=type)

    # ALLOCATE_DIRECTIVE "(" var_list ")" _allocate_clause_list?
    @v_args(inline=True, meta=True)
    def allocate_directive(
        self, meta: Meta, token: Token,
        var_list: list[tree.PyName],
        *clause_list: tree.Clause,
    ) -> tree.DeclareMapper:
        return self._construct(meta, token, clause_list, targets=var_list)

    # DECLARE_VARIANT_DIRECTIVE "(" [py_expr ":"] py_expr ")" _declare_variant_clause_list
    @v_args(inline=True, meta=True)
    def declare_variant_directive(
        self, meta: Meta, token: Token,
        base_name: tree.PyExpr|None, variant_name: tree.PyExpr,
        *clause_list: tree.Clause,
    ) -> tree.DeclareMapper:
        return self._construct(meta, token, clause_list, base_name=base_name, variant_name=variant_name)

    # DECLARE_SIMD_DIRECTIVE ["(" py_expr ")"] _declare_simd_clause_list?
    @v_args(inline=True, meta=True)
    def declare_simd_directive(
        self, meta: Meta, token: Token,
        proc_name: tree.PyExpr|None,
        *clause_list: tree.Clause,
    ) -> tree.DeclareMapper:
        return self._construct(meta, token, clause_list, proc_name=proc_name)

    # DECLARE_TARGET_DIRECTIVE "(" var_list ")" -> declare_target_directive
    @v_args(inline=True, meta=True)
    def declare_target_directive(
        self, meta: Meta, token: Token, var_list: list[tree.PyName]
    ) -> tree.DeclareMapper:
        return self._construct(meta, token, [], targets=var_list)

    # CRITICAL_DIRECTIVE ["(" IDENTIFIER ")" [","? hint_clause]]
    @v_args(inline=True, meta=True)
    def critical_directive(
        self, meta: Meta, token: Token, name: tree.PyName, hint: tree.Hint|None
    ) -> tree.Critical:
        return self._construct(meta, token, [hint] if hint is not None else [], critical_name=name)

    # FLUSH_DIRECTIVE [acq_rel_clause | acquire_clause | ...] ["(" var_list ")"]
    @v_args(inline=True, meta=True)
    def flush_directive(
        self, meta: Meta, token: Token,
        clause: tree.Clause|None, var_list: list[tree.PyName]|None
    ) -> tree.Flush:
        return self._construct(meta, token, [clause] if clause is not None else [], targets=var_list)

    # DEPOBJ_DIRECTIVE "(" IDENTIFIER ")" (destroy_clause | init_clause | depobj_update_clause)
    @v_args(inline=True, meta=True)
    def depobj_directive(
        self, meta: Meta, token: Token,
        object: tree.PyName, clause: tree.Clause
    ) -> tree.Depobj:
        return self._construct(meta, token, [clause], object=object)

    # CANCEL_DIRECTIVE [directive_name ":"] construct_type_clause [","? if_clause]
    @v_args(inline=True, meta=True)
    def cancel_directive(
        self, meta: Meta, token: Token,
        _directive_name: tree.DirectiveName|None, construct_type: Token, if_: tree.If|None
    ) -> tree.Cancel:
        # TODO: ignoring directive_name here
        return tree.Cancel(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            nconstruct_type = self._name_from_token(construct_type),
            if_ = if_
        )

    # CANCELLATION_POINT_DIRECTIVE [directive_name ":"] construct_type_clause
    @v_args(inline=True, meta=True)
    def cancellation_point_directive(
        self, meta: Meta, token: Token,
        _directive_name: tree.DirectiveName|None, construct_type: Token
    ) -> tree.CancellationPoint:
        # TODO: ignoring directive_name here
        return tree.CancellationPoint(
            span = self.sv.meta2span(meta),
            name = self._name_from_token(token),
            nconstruct_type = self._name_from_token(construct_type),
        )

    ############################################################################

    # combined_directive: combined_directive_name combined_directive_name+ [combined_clause_list]
    def combined_directive(self, node: Tree) -> tree.Directive:
        span = self.sv.meta2span(node.meta)
        clause_list = (
            cast("list[tree.Clause]", node.children[-1].children)
            if isinstance(node.children[-1], Tree)
            else []
        )

        constructs = {}
        for directive_token in node.children[:-1]:
            directive_token = cast("Token", directive_token)
            r: tuple[tree.Construct, list[tree.Clause]] = self._construct_with_rejected(node.meta, directive_token, clause_list)
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

        # Otherwise, assuming this rule format, process the directive in a general case:
        #   task_directive: TASK_DIRECTIVE _task_clause_list?

        directive_token = cast("Token", directive.children[0])
        clause_list     = cast("list[tree.Clause]", directive.children[1:])
        construct: tree.Construct = self._construct(directive.meta, directive_token, clause_list)

        return tree.Directive(
            span = self.sv.meta2span(directive.meta),
            string = self.sv.source_text(self.sv.span), # NOTE: this includes the quotes
            constructs = {construct.id: construct},
        )
