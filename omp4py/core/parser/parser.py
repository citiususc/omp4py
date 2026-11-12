"""Parsing utilities for OpenMP-like directives in the `omp4py` preprocessor.

This module extracts directive text from Python AST nodes, parses the
directive syntax, and transforms the resulting parse tree into structured
directive objects.

It also preserves source locations throughout parsing so that syntax errors
can report the corresponding positions in the original Python source.

The main entry points are:
    extract_directive: Recover directive text from a Python string constant.
    parse_directive: Parse directive text into a structured representation.
    syntax_error: Create a source-aware syntax error.
"""

from __future__ import annotations

import ast
import textwrap
import typing
from collections import defaultdict
from collections.abc import Iterator
from io import StringIO

from omp4py.core.parser import openmp_lark, string_lark, tree
from omp4py.core.parser.clause_assignment import assign_clauses
from omp4py.core.parser.compound import DIRECTIVES
from omp4py.core.parser.error import parser_error, syntax_error
from omp4py.core.parser.openmp_lark import (
    ContextualLexer,
    Lexer,
    LexerState,
    Meta,
    ParserState,
    Token,
    v_args,
)
from omp4py.core.parser.tree import Clause, Directive, Modifier, Name, PyExpr, PyName, Span

__all__ = ["extract_directive", "parse_directive", "syntax_error"]


string_parser: string_lark.Lark = string_lark.Lark_StandAlone()
openmp_parser: openmp_lark.Lark = openmp_lark.Lark_StandAlone(propagate_positions=True)


def extract_directive(node: ast.Constant, full_source: str, filename: str) -> str:
    """Extract the raw directive source from an AST constant node.

    This function retrieves the textual representation of a directive
    stored inside a string constant, preserving the original source
    positions possible.

    Simple string literals are returned directly from their evaluated
    value. More complex string forms, such as multiline literals or
    escaped strings, require recovering the original source segment in
    order to preserve accurate token positioning during parsing.

    Args:
        node (ast.Constant): AST constant containing the directive string.
        full_source (str): Complete source code being processed.
        filename (str): Source filename used for error reporting.

    Returns:
        str: Directive text with source layout preserved where required.

    Raises:
        SyntaxError: If the directive's source segment cannot be recovered.
    """
    node_value = str(node.value)
    raw_source: str | None = ast.get_source_segment(full_source, node)
    if raw_source is None:
        msg = "source directive not found"
        raise syntax_error(msg, Span.from_ast(node), full_source, filename)

    if len(raw_source) - 2 == len(node_value):
        return " " + node_value + " "

    return StringTransformer().transform(string_parser.parse(raw_source))


def parse_directive(code: str, span: Span, full_source: str, filename: str) -> Directive:
    """Parse a directive string into a structured directive tree.

    This function performs the parsing stage for OpenMP-like directives,
    converting the textual directive representation into a structured
    syntax tree composed of constructs, clauses, and modifiers.

    The resulting `Directive` object is later consumed by the
    transformation pipeline to apply AST-level code generation and
    rewriting.

    Parsing preserves source location information to enable accurate
    diagnostics and contextual error reporting during later stages.

    Args:
        code (str): Directive text to parse.
        span (Span): Source span of the directive in the original Python code.
        full_source (str): Complete source code being processed.
        filename (str): Source filename used for error reporting.

    Returns:
        Directive: Structured representation of the parsed directive.

    Raises:
        SyntaxError: If the directive contains invalid syntax or semantics.
    """
    try:
        lark_tree: openmp_lark.Tree = openmp_parser.parse(code)
        return OpenmpTransformer(code, span, full_source, filename).transform(lark_tree)
    except openmp_lark.UnexpectedInput as ex:
        raise parser_error(openmp_parser, ex, span, full_source, filename) from None
    except openmp_lark.VisitError as ex:
        raise ex.orig_exc from None


@v_args(inline=True)
class StringTransformer(string_lark.Transformer):
    """Reconstruct string-literal contents while preserving source layout.

    Transforms tokens from the string-literal parser into text whose line
    and column positions correspond to the original literal's contents.
    This allows subsequent directive parsing to produce accurate locations.
    """

    @staticmethod
    def start(*strings: string_lark.Token) -> str:
        """Reconstruct string contents from parsed literal tokens.

        Preserves line breaks and column spacing while removing the literal
        delimiters and accounting for prefixes such as string markers.

        Args:
            *strings (string_lark.Token): Tokens representing the string
                literal's components.

        Returns:
            str: Reconstructed string content with source positions preserved.

        Raises:
            RuntimeError: If a token lacks the required source-position data.
        """
        result = StringIO()
        last_line: int = 1
        last_col: int = 1
        for token in strings:
            if token.line is None or token.column is None or token.end_line is None or token.end_column is None:
                msg = "Lark token must have line and column"
                raise RuntimeError(msg)  # never raise
            value = str(token)
            line: int = token.line
            end_line: int = token.end_line
            col: int = token.column
            end_col: int = token.end_column

            if last_line < line:
                result.write("\n" * (line - last_line))
                last_col = 1
            if last_col < col:
                result.write(" " * (col - last_col))

            while value[0].isalpha():
                result.write(" ")
                value = value[1:]

            if value.startswith(('"""', "'''")):
                result.write("   " + value[3:-3] + "   ")
            elif line == end_line:
                result.write(" " + value[1:-1] + " ")
            else:
                start, *middle, end = value.splitlines()
                result.write(" " + start[1:-1] + " \n")
                result.writelines([s[:-1] + " \n" for s in middle])
                result.write(end[:-1] + " ")

            last_line = end_line
            last_col = end_col

        result.seek(0)
        return result.read()


type Modifiers = list[Modifier] | None


class OpenmpContextualLexer(Lexer):
    """Extend Lark's contextual lexer to recognize modifier separators.

    Tracks nested delimiters and predicts whether a colon at the relevant
    nesting level separates a clause or construct from its modifiers.
    When appropriate, injects the `_HM` token expected by the grammar.

    Attributes:
        lexer (ContextualLexer): The underlying contextual lexer.
    """

    lexer: ContextualLexer

    def __init__(self, lexer: Lexer) -> None:
        """Initialize the lexer wrapper.

        Args:
            lexer (Lexer): Lexer to wrap. Must be a `ContextualLexer`.

        Raises:
            TypeError: If `lexer` is not a `ContextualLexer`.
        """
        if not isinstance(lexer, ContextualLexer):
            msg = "contextual lexer required"
            raise TypeError(msg)
        self.lexer = lexer

    def lex(self, lexer_state: LexerState, parser_state: ParserState) -> Iterator[Token]:
        """Yield tokens while detecting modifier separators.

        Tracks nesting of parentheses, brackets, and braces. After an
        opening delimiter at the outermost level, checks whether a modifier
        separator is present and emits the corresponding marker token.

        Args:
            lexer_state (LexerState): Current lexer state.
            parser_state (ParserState): Current parser state.

        Yields:
            Token: Tokens from the underlying lexer, including injected
                modifier markers when detected.
        """
        level = 0
        for token in self.lexer.lex(lexer_state, parser_state):
            if token.value in ("(", "[", "{"):
                level += 1
                yield token
                if level == 1 and self.predict_modifiers(lexer_state, parser_state):
                    yield Token.new_borrow_pos("_HM", "", token)
                continue
            elif token.value in (")", "]", "}"):
                level -= 1
            yield token

    def predict_modifiers(self, lexer_state: LexerState, parser_state: ParserState) -> bool:
        """Determine whether the current parenthesized region has modifiers.

        Scans ahead using a copied lexer state and the root lexer. Returns
        `True` when it finds a colon at the top level of the region, and
        `False` when the region closes or the input ends first.

        Args:
            lexer_state (LexerState): Current lexer state used as the
                starting point for lookahead.
            parser_state (ParserState): Current parser state.

        Returns:
            bool: Whether a top-level modifier separator is present.
        """
        new_line_ctr = lexer_state.line_ctr.__new__(lexer_state.line_ctr.__class__)
        for f in lexer_state.line_ctr.__slots__:
            setattr(new_line_ctr, f, getattr(lexer_state.line_ctr, f))

        lexer_state_copy = LexerState(lexer_state.text, new_line_ctr, lexer_state.last_token)
        level = 1
        for token in self.lexer.root_lexer.lex(lexer_state_copy, parser_state):
            if token.value in ("(", "[", "{"):
                level += 1
            elif token.value in (")", "]", "}"):
                level -= 1
                if level == 0:
                    return False
            elif level == 1 and token.value == ":":
                return True

        return False


openmp_parser.parser.lexer = OpenmpContextualLexer(typing.cast("Lexer", openmp_parser.parser.lexer))


@v_args(inline=True, meta=True)
class OpenmpTransformer(openmp_lark.Transformer):
    """Convert OpenMP parse trees into structured directive objects.

    Uses parse-tree metadata and the original directive source to construct
    syntax-tree nodes with accurate source spans. Also provides helpers for
    validating and assembling directives, clauses, modifiers, and embedded
    Python expressions.

    Attributes:
        code (str): Directive text being transformed.
        span (tree.Span): Directive's location in the original Python source.
        full_source (str): Complete source code being processed.
        filename (str): Source filename used for error reporting.
    """

    code: str
    span: tree.Span
    full_source: str
    filename: str

    def __init__(self, code: str, span: tree.Span, full_source: str, filename: str) -> None:
        """Initialize the transformer with directive and source context.

        Args:
            code (str): Directive text being transformed.
            span (tree.Span): Directive's original source span.
            full_source (str): Complete source code being processed.
            filename (str): Source filename used for error reporting.
        """
        super().__init__()
        self.code = code
        self.span = span
        self.full_source = full_source
        self.filename = filename

    def token2meta(self, token: Token)-> Meta:
        """Convert Lark token to a lark parse metadata.

        Args:
            token (Token): Lark token for a parsed node.

        Returns:
            tree.Meta: Corresponding lark parse metadata.
        """
        node = openmp_lark.Tree("", [])
        openmp_lark.PropagatePositions(lambda _: node)([token])
        return node.meta


    def meta2span(self, meta: Meta) -> tree.Span:
        """Convert Lark parse metadata to an original-source span.

        Adjusts the parse-tree line and column positions using the directive's
        base location, accounting for the directive's starting column on its
        first line.

        Args:
            meta (Meta): Lark metadata for a parsed node.

        Returns:
            tree.Span: Corresponding span in the original Python source.
        """
        return tree.Span(
            self.span.lineno + meta.line - 1,
            (self.span.offset if meta.line == 1 else 0) + meta.column - 1,
            self.span.lineno + meta.end_line - 1,
            (self.span.offset if meta.line == 1 else 0) + meta.end_column - 1,
        )

    def syntax_error(self, message: str, span: tree.Span) -> SyntaxError:
        """Create a syntax error associated with the original source.

        Args:
            message (str): Error message.
            span (tree.Span): Source span associated with the error.

        Returns:
            SyntaxError: Source-aware syntax error with filename and location.
        """
        return syntax_error(message, span, self.full_source, self.filename)

    ###################################################
    #################### Templates ####################
    ###################################################

    def literal(self, meta: Meta, *_: typing.Any) -> Name:
        """Create a name from its source text.

        Args:
            meta (Meta): Parse metadata identifying the source segment.
            *_ (Any): Additional grammar values, ignored by this helper.

        Returns:
            Name: Name node containing the corresponding source text and span.
        """
        return Name(self.meta2span(meta), self.code[meta.start_pos : meta.end_pos])

    def py_lit(self, meta: Meta) -> PyName:
        """Create a Python-name node from its source text.

        Args:
            meta (Meta): Parse metadata identifying the source segment.

        Returns:
            PyName: Python-name node containing the source text and span.
        """
        return PyName(self.meta2span(meta), self.code[meta.start_pos : meta.end_pos])

    def as_list[T: tree.OmpNode](self, _: Meta, *items: T) -> list[T]:
        """Collect transformed grammar items into a list.

        Args:
            meta (Meta): Parse metadata, unused by this helper.
            *items (T): Transformed syntax-tree nodes.

        Returns:
            list[T]: The supplied items as a list.
        """
        return list(items)

    ###################################################
    #################### Directives ###################
    ###################################################

    def basic_directive(self, meta: Meta, name: Name, clauses: list[Clause]) -> tree.Directive:
        """Build a directive containing a single construct with basic syntax.

        Resolves the construct class from the directive name, creates the
        construct, attaches its clauses, and wraps it in a `Directive` node.

        Args:
            meta (Meta): Parse metadata for the directive.
            name (Name): Parsed directive name.
            clauses (list[Clause]): Clauses associated with the construct.

        Returns:
            tree.Directive: Directive containing the constructed node.
        """
        tree_name = "".join(word.capitalize() for word in str(name).replace("_", " ").split())
        tree_cls: type[tree.Construct] = getattr(tree, tree_name)
        construct: tree.Construct = tree_cls(self.meta2span(meta), name)
        return tree.Directive(construct.span, self.code, [self.set_all_clauses(construct, clauses)])

    def atomic_dir(self, meta: Meta, name: Name, clauses: list[Clause]) -> tree.Directive:
        """Build and validate an atomic directive.

        Attaches clauses to an `Atomic` construct and validates the number of
        clauses against the supported atomic form.

        Args:
            meta (Meta): Parse metadata for the directive.
            name (Name): Atomic directive name.
            clauses (list[Clause]): Clauses associated with the directive.

        Returns:
            tree.Directive: Directive containing the validated atomic construct.

        Raises:
            SyntaxError: If no clause is provided or too many clauses are present.
        """
        atomic = self.set_all_clauses(tree.Atomic(self.meta2span(meta), name), clauses)
        if (atomic.seq_cst is None and len(clauses) == 1) or (atomic.seq_cst and len(clauses) == 2):  # ruff: ignore[magic-value-comparison]
            return tree.Directive(atomic.span, self.code, [atomic])
        if len(clauses) == 0 or (atomic.seq_cst is not None and len(clauses) == 1):
            msg = "atomic require at least one clause"
            raise self.syntax_error(msg, atomic.span)
        msg = "too many atomic clauses"
        raise self.syntax_error(msg, [c for c in clauses if c != atomic.seq_cst][1].span)

    def thprivate_dir(self, meta: Meta, name: Name, targets: list[PyName], clauses: list[Clause]) -> tree.Directive:
        """Build a threadprivate directive.

        Creates a `ThreadPrivate` construct for the specified targets, attaches
        its clauses, and wraps it in a directive node.

        Args:
            meta (Meta): Parse metadata for the directive.
            name (Name): Directive name.
            targets (list[PyName]): Python names targeted by the directive.
            clauses (list[Clause]): Clauses associated with the directive.

        Returns:
            tree.Directive: Directive containing the threadprivate construct.
        """
        tp = self.set_all_clauses(tree.ThreadPrivate(self.meta2span(meta), name, targets), clauses)
        return tree.Directive(tp.span, self.code, [tp])

    def cancel_dir(self, meta: Meta, name: Name, cons: Name, clauses: list[Clause]) -> tree.Directive:
        """Build a cancellation directive for a specified construct.

        Args:
            meta (Meta): Parse metadata for the directive.
            name (Name): Cancellation directive name.
            cons (Name): Name of the construct being cancelled.
            clauses (list[Clause]): Clauses associated with the directive.

        Returns:
            tree.Directive: Directive containing the `Cancel` construct.
        """
        cp = tree.ConstructType(cons.span, name, cons.string)
        cancel = self.set_all_clauses(tree.Cancel(self.meta2span(meta), name, cp), clauses)
        return tree.Directive(cancel.span, self.code, [cancel])

    def can_point_dir(self, meta: Meta, name: Name, cons: Name, clauses: list[Clause]) -> tree.Directive:
        """Build a cancellation-point directive.

        Args:
            meta (Meta): Parse metadata for the directive.
            name (Name): Cancellation-point directive name.
            cons (Name): Name of the construct associated with the point.
            clauses (list[Clause]): Clauses associated with the directive.

        Returns:
            tree.Directive: Directive containing the cancellation-point construct.
        """
        cp = tree.ConstructType(cons.span, name, cons.string)
        cancel = self.set_all_clauses(tree.CancellationPoint(self.meta2span(meta), name, cp), clauses)
        return tree.Directive(cancel.span, self.code, [cancel])

    def dec_reduction_dir(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        meta: Meta,
        name: Name,
        op: PyName,
        ann_list: list[tree.PyExpr],
        old_comb: tree.Combiner | None,
        clauses: list[Clause],
    ) -> tree.Directive:
        comb = old_comb or self.get_clause(meta, tree.Combiner, clauses)
        dr = self.set_all_clauses(tree.DeclareReduction(self.meta2span(meta), name, op, ann_list, comb), clauses)
        return tree.Directive(dr.span, self.code, [dr])

    ###################################################
    ################### Composition ###################
    ###################################################
    def check_compose_directive(self, directive: tree.Directive) -> tree.Directive:
        count = 0
        match directive.constructs:
            case [*_, tree.For() | tree.Distribute() | tree.Taskloop(), tree.Simd()]:
                count = 2
            case [*_, tree.Distribute(), tree.Parallel(), tree.For()]:
                count = 3
            case [*_, tree.Distribute(), tree.Parallel(), tree.For(), tree.Simd()]:
                count = 4

        for i in range(count):
            directive.constructs[-i - 1].compound_len = i

        return directive

    def compound_directive(self, meta: Meta, names: list[Name], clauses: list[Clause]) -> tree.Directive:
        """Build a compound directive and assign its clauses to its components.

        Creates the individual directives from the provided names, then groups
        the clauses according to their associated directive name. Clauses
        without an explicit directive name are associated with the full
        compound directive name.

        For each clause group, assigns clauses directly when the group matches
        a component name. Otherwise, determines which components are eligible
        to receive the clauses and uses ``assign_clauses`` to distribute them.
        Raises a syntax error if any clause cannot be assigned.

        Finally, constructs the compound directive and validates its
        composition with ``check_compose_directive``.

        Args:
            meta (Meta): Parse metadata for the compound directive.
            names (list[Name]): Names of the directives forming the composition.
            clauses (list[Clause]): Clauses associated with the compound
                directive and its individual components.

        Returns:
            tree.Directive: Constructed and validated compound directive.

        Raises:
            SyntaxError: If a clause is not valid for its associated directive
                or if the resulting compound directive violates composition
                rules.
        """
        constructs = {str(name).lower(): self.basic_directive(meta, name, []).constructs[0] for name in names}
        full_comp_name = " ".join(name.string for name in names).lower()
        named_clauses: dict[str, list[Clause]] = defaultdict(list)
        for c in clauses:
            named_clauses[str(c.directive_name).lower() if c.directive_name else full_comp_name].append(c)

        for group_name, group_clauses in named_clauses.items():
            if group_name in constructs:
                self.set_all_clauses(constructs[group_name], group_clauses)
                continue
            allowed_constructs = {name: cons for name, cons in constructs.items() if name in group_name}
            assignment = assign_clauses(allowed_constructs, group_clauses)
            [self.set_all_clauses(constructs[name],acls) for name, acls in assignment.items()]

            if orphan:= set(group_clauses).difference(*[ set(acls) for acls in assignment.values()]):
                first = next(iter(orphan))
                msg = f"'{first.cid()}' is not valid for '{group_name}'"
                raise self.syntax_error(msg, first.span)


        return self.check_compose_directive(tree.Directive(self.meta2span(meta), self.code, list(constructs.values())))

    def comp_constructs(self, meta: Meta, *names: Name) -> list[Name]:
        """Validate and combine the names of a composed directive.

        Joins the provided directive names and checks whether the resulting
        combination is included in ``DIRECTIVES``. If the combination is
        invalid, progressively removes trailing names to identify the valid
        prefix and reports the first name that cannot be composed with it.

        Args:
            meta (Meta): Parse metadata for the composed directive.
            *names (Name): Parsed names forming the directive composition.

        Returns:
            list[Name]: The validated directive names, preserving their
                original order.

        Raises:
            SyntaxError: If the directive combination is not valid. The error
                identifies the first directive name that cannot be appended
                to the preceding valid composition.
        """
        full_comp_name = " ".join(name.string for name in names)

        if full_comp_name.upper() not in DIRECTIVES:
            directives = sorted(DIRECTIVES)
            valid_name = full_comp_name.upper()
            last = len(names) - 1
            while valid_name:
                if [directive for directive in directives if valid_name in directives]:
                    break
                valid_name, *rest = valid_name.rsplit(" ", 1)
                if not rest:
                    break
                last -= 1
            valid_name = " ".join(name.string for name in names[: last + 1])
            msg = f"'{names[last + 1]}' is not valid for '{valid_name}'"
            raise self.syntax_error(msg, names[last + 1].span)
        return list(names)

    ###################################################
    ##################### Clauses #####################
    ###################################################

    def set_clauses[T: tree.OmpNode](self, node: T, clauses: list[Clause], final: bool = True) -> list[Clause]:
        """Attach clauses to a syntax-tree node.

        Each clause is assigned to the corresponding attribute on `node`.
        Attributes that hold lists accept multiple clauses; single-valued
        attributes reject duplicate clauses. If `final` is false, clauses
        unsupported by the node are returned as remaining clauses.

        Args:
            node (T): Construct or syntax-tree node receiving the clauses.
            clauses (list[Clause]): Clauses to attach.
            final (bool): Whether unsupported clauses should raise an error
                instead of being returned.

        Returns:
            list[Clause]: Clauses not attached to the node. Empty when `final`
                is true and all clauses are valid.

        Raises:
            SyntaxError: If a clause is unsupported or duplicated.
        """
        remainder: list[Clause] = []
        for cl in clauses:
            if not hasattr(node, cl.id()):
                if not final:
                    remainder.append(cl)
                    continue
                msg = f"'{cl.cid()}' is not valid for '{node.cid()}'"
                raise self.syntax_error(msg, cl.span)
            match v := getattr(node, cl.id()):
                case None:
                    setattr(node, cl.id(), cl)
                case list():
                    v.append(cl)
                case _:
                    if id(v) != id(cl):
                        msg = f"too many '{cl.cid()}' clauses"
                        raise self.syntax_error(msg, cl.span)
        return remainder

    def set_all_clauses[T: tree.OmpNode](self, node: T, clauses: list[Clause]) -> T:
        """Attach all clauses to a node and return the node.

        Args:
            node (T): Construct or syntax-tree node receiving the clauses.
            clauses (list[Clause]): Clauses to attach.

        Returns:
            T: The same node, with its clauses assigned.
        """
        self.set_clauses(node, clauses)
        return node

    def get_clause[T: Clause](self, meta: Meta, tp: type[T], clauses: list[Clause]) -> T:
        """Retrieve the first clause of a specified type.

        Args:
            meta (Meta): Parse metadata used to locate a missing-clause error.
            tp (type[T]): Clause type to search for.
            clauses (list[Clause]): Clauses to search.

        Returns:
            T: The first clause matching `tp`.

        Raises:
            SyntaxError: If no clause of the requested type is present.
        """
        for clause in clauses:
            if isinstance(clause, tp):
                return clause
        msg = f"clause '{tp.id()}' required"
        raise self.syntax_error(msg, self.meta2span(meta))

    def clause(self, meta: Meta, name: Name, *args: typing.Any) -> list[Clause]:  # ruff: ignore[any-type]
        """Dispatch clause construction to the corresponding clause handler.

        Resolves the handler method from the clause name and passes the parsed
        arguments to it. The handler constructs the clause node and applies any
        associated modifiers.

        Args:
            meta (Meta): Parse metadata for the clause.
            name (Name): Parsed clause name used to select its handler.
            *args (Any): Additional parsed arguments required by the handler.

        Returns:
            list[Clause]: Clause node or nodes produced by the selected handler.
        """
        return getattr(self, f"{name}_cl")(meta, name, *args)

    def if_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.If:
        return self.set_mods(tree.If(self.meta2span(meta), name, expr), mods)

    def num_threads_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.NumThreads:
        return self.set_mods(tree.NumThreads(self.meta2span(meta), name, expr), mods)

    def default_cl(self, meta: Meta, name: Name, mods: Modifiers, data: Name) -> tree.Default:
        return self.set_mods(tree.Default(self.meta2span(meta), name, tree.DefaultData(data.span, data.string)), mods)

    def private_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Private:
        return self.set_mods(tree.Private(self.meta2span(meta), name, targets), mods)

    def firstprivate_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.FirstPrivate:
        return self.set_mods(tree.FirstPrivate(self.meta2span(meta), name, targets), mods)

    def lastprivate_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.LastPrivate:
        return self.set_mods(tree.LastPrivate(self.meta2span(meta), name, targets), mods)

    def copyprivate_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.CopyPrivate:
        return self.set_mods(tree.CopyPrivate(self.meta2span(meta), name, targets), mods)

    def shared_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Shared:
        return self.set_mods(tree.Shared(self.meta2span(meta), name, targets), mods)

    def copyin_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.CopyIn:
        return self.set_mods(tree.CopyIn(self.meta2span(meta), name, targets), mods)

    def reduction_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Reduction:
        op = self.get_mod(meta, tree.ReductionOp, mods)
        return self.set_mods(tree.Reduction(self.meta2span(meta), name, targets, op), mods)

    def proc_bind_cl(self, meta: Meta, name: Name, mods: Modifiers, arg: PyName) -> tree.ProcBind:
        affinity = tree.AffinityPolicy(arg.span, arg.string)
        return self.set_mods(tree.ProcBind(self.meta2span(meta), name, affinity), mods)

    def linear_cl_old(
        self,
        meta: Meta,
        name: Name,
        mod: Modifier,
        targets: list[PyName],
        mods: Modifiers,
    ) -> tree.Linear:
        return self.linear_cl(meta, name, targets, [mod, *(mods or [])])

    def linear_cl(self, meta: Meta, name: Name, targets: list[PyName], mods: Modifiers) -> tree.Linear:
        return self.set_mods(tree.Linear(self.meta2span(meta), name, targets), mods)

    def schedule_cl(
        self,
        meta: Meta,
        name: Name,
        mods: Modifiers,
        kind: Name,
        expr: tree.PyExpr | None = None,
    ) -> tree.Schedule:
        sc = tree.Schedule(self.meta2span(meta), name, tree.ScheduleType(kind.span, kind.string))
        sc.chunk_size = expr
        return self.set_mods(sc, mods)

    def collapse_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Collapse:
        n = self.pyint(expr)
        if n is None:
            msg = "collapse argument needs positive constant integer expression"
            raise self.syntax_error(msg, expr.span)
        return self.set_mods(tree.Collapse(self.meta2span(meta), name, n), mods)

    def orderer_cl(
        self,
        meta: Meta,
        name: Name,
        mods: Modifiers,
        expr: tree.PyExpr | None = None,
    ) -> tree.OrderedClause:
        n = None
        if expr is not None:
            n = self.pyint(expr)
            if n is None:
                msg = "ordered argument needs positive constant integer expression"
                raise self.syntax_error(msg, expr.span)
        return self.set_mods(tree.OrderedClause(self.meta2span(meta), name, n), mods)

    def nowait_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr | None = None) -> tree.NoWait:
        return self.set_mods(tree.NoWait(self.meta2span(meta), name, expr), mods)

    def safelen_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.SafeLen:
        return self.set_mods(tree.SafeLen(self.meta2span(meta), name, expr), mods)

    def simdlen_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Simdlen:
        return self.set_mods(tree.Simdlen(self.meta2span(meta), name, expr), mods)

    def aligned_cl(self, meta: Meta, name: Name, targets: list[PyName], mods: Modifiers) -> tree.Aligned:
        return self.set_mods(tree.Aligned(self.meta2span(meta), name, targets), mods)

    def uniform_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Uniform:
        return self.set_mods(tree.Uniform(self.meta2span(meta), name, targets), mods)

    def inbranch_cl(self, meta: Meta, name: Name, mods: Modifiers) -> tree.InBranch:
        return self.set_mods(tree.InBranch(self.meta2span(meta), name), mods)

    def notinbranch_cl(self, meta: Meta, name: Name, mods: Modifiers) -> tree.NotInBranch:
        return self.set_mods(tree.NotInBranch(self.meta2span(meta), name), mods)

    def final_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Final:
        return self.set_mods(tree.Final(self.meta2span(meta), name, expr), mods)

    def untied_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr | None = None) -> tree.Untied:
        return self.set_mods(tree.Untied(self.meta2span(meta), name, expr), mods)

    def mergeable_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Mergeable:
        return self.set_mods(tree.Mergeable(self.meta2span(meta), name, expr), mods)

    def depend_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Depend:
        tp = self.get_mod(meta, tree.DependenceType, mods)
        return self.set_mods(tree.Depend(self.meta2span(meta), name, tp, targets), mods)

    def priority_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Priority:
        return self.set_mods(tree.Priority(self.meta2span(meta), name, expr), mods)

    def grainsize_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.GrainSize:
        return self.set_mods(tree.GrainSize(self.meta2span(meta), name, expr), mods)

    def num_tasks_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.NumTasks:
        return self.set_mods(tree.NumTasks(self.meta2span(meta), name, expr), mods)

    def nogroup_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr | None = None) -> tree.NoGroup:
        return self.set_mods(tree.NoGroup(self.meta2span(meta), name, expr), mods)

    def device_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Device:
        return self.set_mods(tree.Device(self.meta2span(meta), name, expr), mods)

    def map_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Map:
        tp = self.get_mod(meta, tree.MapType, mods)
        return self.set_mods(tree.Map(self.meta2span(meta), name, targets, tp), mods)

    def use_device_ptr_cl(
        self,
        meta: Meta,
        name: Name,
        mods: Modifiers,
        targets: list[PyName],
    ) -> tree.UseDevicePtr:
        return self.set_mods(tree.UseDevicePtr(self.meta2span(meta), name, targets), mods)

    def defaultmap_cl(self, meta: Meta, name: Name, ndb: tree.PyName, mods: Modifiers) -> tree.DefaultMap:
        db = tree.DefaultBehavior(ndb.span, ndb.string)
        return self.set_mods(tree.DefaultMap(self.meta2span(meta), name, db), mods)

    def to_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.To:
        return self.set_mods(tree.To(self.meta2span(meta), name, targets), mods)

    def from_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.From:
        return self.set_mods(tree.From(self.meta2span(meta), name, targets), mods)

    def link_cl(self, meta: Meta, name: Name, mods: Modifiers, targets: list[PyName]) -> tree.Link:
        return self.set_mods(tree.Link(self.meta2span(meta), name, targets), mods)

    def num_teams_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.NumTeams:
        return self.set_mods(tree.NumTeams(self.meta2span(meta), name, expr), mods)

    def thread_limit_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.ThreadLimit:
        return self.set_mods(tree.ThreadLimit(self.meta2span(meta), name, expr), mods)

    def dist_schedule_cl(
        self,
        meta: Meta,
        name: Name,
        mods: Modifiers,
        kind: Name,
        expr: tree.PyExpr | None = None,
    ) -> tree.DistSchedule:
        sc = tree.DistSchedule(self.meta2span(meta), name, tree.DistScheduleType(kind.span, kind.string))
        sc.chunk_size = expr
        return self.set_mods(sc, mods)

    def seq_cst_cl(self, meta: Meta, name: Name, mods: Modifiers) -> tree.SeqCst:
        return self.set_mods(tree.SeqCst(self.meta2span(meta), name), mods)

    def read_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Read:
        return self.set_mods(tree.Read(self.meta2span(meta), name, expr), mods)

    def write_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Write:
        return self.set_mods(tree.Write(self.meta2span(meta), name, expr), mods)

    def update_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Update:
        return self.set_mods(tree.Update(self.meta2span(meta), name, expr), mods)

    def capture_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Capture:
        return self.set_mods(tree.Capture(self.meta2span(meta), name, expr), mods)

    def filter_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Filter:
        return self.set_mods(tree.Filter(self.meta2span(meta), name, expr), mods)

    def combiner_cl(self, meta: Meta, name: Name, mods: Modifiers, code: tree.PyCode) -> tree.Combiner:
        return self.set_mods(tree.Combiner(self.meta2span(meta), name, code), mods)

    def initializer_cl(self, meta: Meta, name: Name, mods: Modifiers, code: tree.PyCode) -> tree.Initializer:
        return self.set_mods(tree.Initializer(self.meta2span(meta), name, code), mods)

    ###################################################
    #################### Modifiers ####################
    ###################################################

    def set_mods[T: tree.OmpNode](self, node: T, mods: Modifiers) -> T:
        """Attach modifiers to a syntax-tree node.

        Assigns each modifier to the corresponding attribute on `node`.
        Attributes that hold lists accept multiple modifiers, while
        single-valued attributes reject duplicates.

        Args:
            node (T): Construct or syntax-tree node receiving the modifiers.
            mods (Modifiers): Modifiers to attach, or `None` if there are none.

        Returns:
            T: The same node, with its modifiers assigned.

        Raises:
            SyntaxError: If a modifier is unsupported by the node or duplicated.
        """
        if mods is not None:
            for mod in mods:
                if not hasattr(node, mod.id()):
                    msg = f"'{mod.cid()}' is not a valid modifier for '{mod.cid()}'"
                    raise self.syntax_error(msg, mod.span)
                match v := getattr(node, mod.id()):
                    case None:
                        setattr(node, mod.id(), mod)
                    case list():
                        v.append(mod)
                    case _:
                        if id(v) != id(mod):
                            msg = f"too many '{mod.cid()}' modifiers"
                            raise self.syntax_error(msg, mod.span)
        return node

    def get_mod[T: Modifier](self, meta: Meta, tp: type[T], mods: Modifiers) -> T:
        """Retrieve the first modifier of a specified type.

        Searches the supplied modifiers for an instance of `tp`.

        Args:
            meta (Meta): Parse metadata used to locate a missing-modifier error.
            tp (type[T]): Modifier type to search for.
            mods (Modifiers): Modifiers to search, or `None` if there are none.

        Returns:
            T: The first modifier matching `tp`.

        Raises:
            SyntaxError: If no modifier of the requested type is present.
        """
        if mods is not None:
            for mod in mods:
                if isinstance(mod, tp):
                    return mod
        msg = f"modifier '{tp.id()}' required"
        raise self.syntax_error(msg, self.meta2span(meta))

    def directive_mod(self, meta: Meta, *tokens: Token) -> tree.DirectiveName:
        names = [self.literal(self.token2meta(token)) for token in tokens]
        return tree.DirectiveName(self.meta2span(meta), names)

    def dec_rec_mod_old(self, meta: Meta, code: tree.PyCode) -> tree.Combiner:
        return tree.Combiner(self.meta2span(meta), self.literal(meta), code)

    def reduction_mod_op(self, meta: Meta, op: tree.PyName) -> tree.ReductionOp:
        return tree.ReductionOp(self.meta2span(meta), op)

    def linear_mod_mod(self, _: Meta, name: tree.Name) -> tree.LinearModifier:
        return tree.LinearModifier(name.span, name.string)

    def schedule_mod_ordering(self, _: Meta, name: tree.Name) -> tree.ScheduleOrdering:
        return tree.ScheduleOrdering(name.span, name.string)

    def schedule_mod_chunk(self, _: Meta, name: tree.Name) -> tree.ScheduleChunk:
        return tree.ScheduleChunk(name.span, name.string)

    def depend_mod_type(self, _: Meta, name: tree.Name) -> tree.DependenceType:
        return tree.DependenceType(name.span, name.string)

    def map_mod_type(self, _: Meta, name: tree.Name) -> tree.MapType:
        return tree.MapType(name.span, name.string)

    def map_mod_mod(self, _: Meta, name: tree.Name) -> tree.MapModifier:
        return tree.MapModifier(name.span, name.string)

    def defaultmap_mod_cat(self, _: Meta, name: tree.Name) -> tree.VariableCategory:
        return tree.VariableCategory(name.span, name.string)

    ###################################################
    ##################### Python ######################
    ###################################################

    def pyint(self, expr: tree.PyExpr) -> tree.PyInt | None:
        """Extract an integer constant from a Python expression.

        Returns a `PyInt` only when the expression's AST value is an integer
        constant. Other expression forms are not evaluated.

        Args:
            expr (tree.PyExpr): Python expression to inspect.

        Returns:
            tree.PyInt | None: Integer constant with the expression's span,
                or `None` if the expression is not an integer constant.
        """
        match expr.value:
            case ast.Constant(value=int(py_int)):
                return tree.PyInt(expr.span, py_int)
            case _:
                return None

    def py_name(self, meta: Meta, t: Token) -> tree.PyName:
        """Create a Python-name node from a lexer token.

        Args:
            meta (Meta): Parse metadata for the name.
            t (Token): Token containing the name text.

        Returns:
            tree.PyName: Python-name node with its source span and text.
        """
        return tree.PyName(self.meta2span(meta), str(t))

    def py_name_sec(self, _: Meta, name: tree.PyName, sections: list[tree.PySection]) -> tree.PyName:
        """Attach section specifications to a Python-name node.

        Args:
            _ (Meta): Parse metadata, unused by this helper.
            name (tree.PyName): Name node to update.
            sections (list[tree.PySection]): Sections associated with the name.

        Returns:
            tree.PyName: The updated name node.
        """
        name.sections = sections
        return name

    def py_section(self, meta: Meta, *args: Token | PyExpr) -> tree.PySection:
        """Build a Python section specification.

        Interprets the parsed bounds and length, when present, and constructs
        the corresponding section node. Missing bounds or lengths are represented
        as `None`.

        Args:
            meta (Meta): Parse metadata for the section.
            *args (Token | PyExpr): Parsed section delimiters and expressions.

        Returns:
            tree.PySection: Section node containing the parsed bounds and length.
        """
        span = self.meta2span(meta)
        match tuple(args[1:-1]):
            case (PyExpr() as expr1, Token(), PyExpr() as expr2):
                return tree.PySection(span, expr1, expr2)
            case (Token(), PyExpr() as expr2):
                return tree.PySection(span, None, expr2)
            case (PyExpr() as expr1, Token()):
                return tree.PySection(span, expr1, None)
        return tree.PySection(span)

    def py_expr(self, _: Meta, code: tree.PyCode) -> tree.PyExpr:
        """Convert parsed Python code into an expression node.

        The embedded code must contain only an expression.

        Args:
            _ (Meta): Parse metadata, unused by this helper.
            code (tree.PyCode): Parsed Python code.

        Returns:
            tree.PyExpr: Expression node wrapping the parsed AST expression.

        Raises:
            SyntaxError: If the first statement is not an expression or contains more than one statement.
        """
        match code.value[0]:
            case ast.Expr(expr):
                if len(code.value) == 1:
                    return tree.PyExpr(code.span, expr)
        msg = "expected expression"
        raise self.syntax_error(msg, tree.Span.from_ast(code.value[0]))

    def py_stmt(self, _: Meta, code: tree.PyCode) -> tree.PyStmt:
        """Convert parsed Python code into a single-statement node.

        Args:
            _ (Meta): Parse metadata, unused by this helper.
            code (tree.PyCode): Parsed Python code.

        Returns:
            tree.PyStmt: Node wrapping the single parsed statement.

        Raises:
            SyntaxError: If the code contains more than one statement.
        """
        if len(code.value) > 1:
            msg = "too many statements"
            raise self.syntax_error(msg, tree.Span.from_ast(code.value[1]))
        return tree.PyStmt(code.span, code.value[0])

    def py_code(self, _: Meta, code: tree.Name) -> tree.PyCode:
        """Parse embedded Python code and adjust its source locations.

        Parses the text stored in `code`, then shifts AST line and column
        positions so they refer to the original Python source rather than the
        embedded snippet.

        Args:
            _ (Meta): Parse metadata, unused by this helper.
            code (tree.Name): Node containing the embedded Python source text
                and its original source span.

        Returns:
            tree.PyCode: Parsed Python module body with adjusted source locations.

        Raises:
            SyntaxError: If the embedded Python code cannot be parsed.
        """
        try:
            dedent_code = textwrap.dedent((" " * code.span.offset) + code.string)
            module = ast.parse(dedent_code)
        except SyntaxError as ex:
            ex_span = Span(ex.lineno or 1, ex.offset or 0, ex.end_lineno or 1, ex.end_offset or 0)
            ex_span.lineno += code.span.lineno - 1
            ex_span.end_lineno += code.span.lineno - 1
            ex_span.offset += code.span.offset - 1
            if ex_span.lineno == ex_span.end_lineno:
                ex_span.end_offset += code.span.offset - 1
            msg = f"error parsing python code, {ex.msg}"
            raise self.syntax_error(msg, ex_span) from None
        for node in ast.walk(module):
            if hasattr(node, "lineno"):
                node.lineno += code.span.lineno - 1  # ty: ignore[unsupported-operator]
                if hasattr(node, "end_lineno"):
                    node.end_lineno += code.span.lineno - 1  # ty: ignore[unsupported-operator]
                    if hasattr(node, "end_col_offset"):
                        node.end_col_offset += code.span.offset  # ty: ignore[unsupported-operator]
                if hasattr(node, "col_offset"):
                    node.col_offset += code.span.offset  # ty: ignore[unsupported-operator]

        return tree.PyCode(code.span, module.body)
