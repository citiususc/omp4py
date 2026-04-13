"""AST-based OpenMP directive parsing and transformation for `omp4py`.

This module implements the core transformation pipeline of the `omp4py`
preprocessor. It performs a two-phase traversal of the Python AST to
identify, parse, and transform OpenMP-like directives into executable
runtime code.

In the first phase, directive expressions (e.g., `omp("parallel")`) are
located and parsed into structured representations using the parser.

In the second phase, the AST is transformed by applying registered
construct handlers to nodes associated with parsed directives. Each
construct is dispatched through the `construct` function, enabling
extensible transformations.

If a construct is not registered, a syntax error is raised.
"""

from __future__ import annotations

import ast
import typing
from functools import singledispatch

from omp4py.core.parser import parse, syntax_error
from omp4py.core.parser.tree import Construct, Span
from omp4py.core.preprocessor.transformers.context import Context
from omp4py.core.preprocessor.transformers.utils import is_unpack_if

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options

__all__ = ["Context", "OmpTransformer", "construct", "syntax_error_ctx"]


def syntax_error_ctx(message: str, span: Span, ctx: Context) -> SyntaxError:
    """Create a syntax error associated with the current transformation context.

    This helper adds source and filename information to parser errors from the active
    context, ensuring consistent error reporting.

    Args:
        message (str): Error message.
        span (Span): Source span where the error occurred.
        ctx (Context): Active transformation context.

    Returns:
        SyntaxError: Constructed syntax error.
    """
    return syntax_error(message, span, ctx.full_source, ctx.filename)


@singledispatch
def construct(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    """Dispatch transformation for a parsed construct.

    This function acts as the central dispatch mechanism for all OpenMP-like
    constructs. Each construct type must register an implementation that
    transforms the associated AST body.

    If no implementation is registered for the given construct, a syntax
    error is raised.

    Args:
        ctr (Construct): Parsed construct representation.
        body (list[ast.stmt]): AST statements associated with the construct.
        ctx (Context): Active transformation context.

    Returns:
        list[ast.stmt]: Transformed AST statements.

    Raises:
        SyntaxError: If the construct is not implemented.
    """
    msg: str = f"'{ctr.name}' not implemented"
    raise syntax_error_ctx(msg, ctr.span, ctx)


def find_decorator(ctx: Context, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
    """Validate and remove the OpenMP decorator from a definition.

    This function ensures that the decorator associated with the OpenMP
    alias is correctly positioned as the innermost decorator. If valid,
    it is removed from the node to prevent further processing.

    Args:
        ctx (Context): Active transformation context.
        node (ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            Definition node to inspect.

    Raises:
        SyntaxError: If the decorator order is invalid.
    """
    child: ast.expr
    for i, child in enumerate(node.decorator_list):
        match child:
            case (
                ast.Name(id=ctx.opt.alias)
                | ast.Attribute(attr=ctx.opt.alias)
                | ast.Call(ast.Name(id=ctx.opt.alias))
                | ast.Call(ast.Attribute(attr=ctx.opt.alias))
            ):
                if i != len(node.decorator_list) - 1:
                    msg = f"Invalid decorator order: expected the '{ctx.opt.alias}' decorator to be the innermost."
                    raise syntax_error_ctx(msg, Span.from_ast(child), ctx)
                node.decorator_list.pop(i)


class OmpParser(ast.NodeVisitor):
    """AST visitor for directive detection and parsing.

    This visitor performs the first phase of the transformation pipeline,
    identifying directive expressions and parsing them into structured
    representations.

    Detected directives are stored in the transformation context and
    associated with their corresponding AST nodes for later processing
    during the transformation phase.

    Attributes:
        ctx (Context): Active transformation context.
    """
    ctx: Context

    def __init__(self, ctx: Context) -> None:
        """Initialize the parser visitor.

        Args:
            ctx (Context): Active transformation context.
        """
        self.ctx = ctx

    def visit(self, node: ast.AST) -> ast.AST:
        """Visit an AST node while maintaining the node stack.

        This method tracks the traversal path by pushing and popping nodes
        from the context stack, enabling parent-aware analysis.

        Args:
            node (ast.AST): Node to visit.

        Returns:
            ast.AST: Visited node.
        """
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process a function definition.

        Validates and removes the OpenMP decorator if present, then
        continues traversal.

        Args:
            node (ast.FunctionDef): Function definition node.
        """
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process an asynchronous function definition.

        Validates and removes the OpenMP decorator if present, then
        continues traversal.

        Args:
            node (ast.AsyncFunctionDef): Async function definition node.
        """
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process a class definition.

        Validates and removes the OpenMP decorator if present, then
        continues traversal.

        Args:
            node (ast.ClassDef): Class definition node.
        """
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: C901
        """Process a potential directive call.

        This method detects calls to the OpenMP alias (e.g., `omp("...")`),
        validates their structure, and parses the directive string into
        a structured representation.

        The parsed directive is then associated with its corresponding
        AST node (either a statement or a `with` block) and stored in
        the context.

        Args:
            node (ast.Call): Call node to inspect.

        Raises:
            SyntaxError: If the directive usage is invalid.
        """
        match node.func:
            case ast.Name(id=self.ctx.opt.alias) | ast.Attribute(attr=self.ctx.opt.alias):
                pass
            case _:
                return
        alias = self.ctx.opt.alias

        if len(node.keywords) > 0:
            msg = f"{alias} () takes no keyword arguments"
            raise syntax_error_ctx(msg, Span.from_ast(node), self.ctx)

        if len(node.args) != 1:
            msg = f"{alias} () takes exactly one argument ({len(node.args)} given)"
            raise syntax_error_ctx(msg, Span.from_ast(node), self.ctx)

        expr: ast.expr = node.args[0]
        if not isinstance(expr, ast.Constant) or not isinstance(expr.value, str):
            msg = f"{alias} () argument needs constant string expression"
            raise syntax_error_ctx(msg, Span.from_ast(node), self.ctx)

        parent: ast.AST = self.ctx.node_stack[-2]
        grand: ast.AST = self.ctx.node_stack[-3]
        directive_node: ast.Expr | ast.With
        match parent:
            case ast.Expr():
                directive_node = parent
            case ast.withitem() if isinstance(grand, ast.With):
                if len(grand.items) > 1:
                    msg = "cannot use more than one directive item in a 'with' statement"
                    raise syntax_error_ctx(msg, Span.from_ast(node), self.ctx)

                if parent.optional_vars is not None:
                    msg = "'as' is not allowed in a directive item"
                    raise syntax_error_ctx(msg, Span.from_ast(parent.optional_vars), self.ctx)

                directive_node = grand
            case _:
                msg = f"{alias} () can only be used as a statement or in a with block"
                raise syntax_error_ctx(msg, Span.from_ast(node), self.ctx)

        raw_directive: str | None = ast.get_source_segment(self.ctx.full_source, expr)
        if raw_directive is None:
            msg = "raw directive source not found"
            raise syntax_error_ctx(msg, Span.from_ast(expr), self.ctx)

        if len(raw_directive) - 2 == len(expr.value):
            directive = parse(self.ctx.filename, f" {expr.value} ", expr.lineno, expr.col_offset, False)
        else:
            directive = parse(self.ctx.filename, raw_directive, expr.lineno, expr.col_offset, True)
        self.ctx.directives[directive_node] = directive


class OmpTransformer(ast.NodeTransformer):
    """AST transformer for applying OpenMP-like constructs.

    This class performs the second phase of the transformation pipeline,
    applying previously parsed directives to the AST.

    For each node associated with a directive, the corresponding construct
    handler is invoked via the `construct` dispatch function, producing
    transformed code.

    The transformer also manages scope transitions and symbol table updates
    to ensure correct variable handling during transformation.

    Attributes:
        ctx (Context): Active transformation context.
    """
    ctx: Context

    def __init__(self, full_source: str, filename: str, module: ast.Module, is_module: bool, opt: Options) -> None:
        """Initialize the transformer.

        Args:
            full_source (str): Original source code.
            filename (str): Source file name.
            module (ast.Module): Root AST node.
            is_module (bool): Whether transforming a full module.
            opt (Options): Transformation options.
        """
        self.ctx = Context(full_source, filename, module, is_module, opt)

    def transform(self) -> ast.Module:
        """Execute the full transformation pipeline.

        This method performs:
        1. Directive parsing using `OmpParser`
        2. AST transformation using registered constructs
        3. Execution of deferred finalizers

        Returns:
            ast.Module: Transformed AST module.
        """
        if len(self.ctx.node_stack) > 0:
            self.ctx.node_stack.pop()
            OmpParser(self.ctx).visit(self.ctx.module)
            self.visit(self.ctx.module)
            [f() for f in self.ctx.finalizers]
        return self.ctx.module

    def visit(self, node: ast.AST) -> ast.AST:
        """Visit a node while updating symbol table and context state.

        This method updates the symbol table before visiting the node
        and maintains the traversal stack.

        Args:
            node (ast.AST): Node to visit.

        Returns:
            ast.AST: Transformed node.
        """
        self.ctx.symtable.update(node)
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def visit_new_scope[T: ast.AST](self, node: T) -> T:
        """Visit a node within a new scope.

        This method creates a new scope for the node, performs traversal,
        and restores the previous scope afterward.

        Args:
            node (ast.AST): Node defining a new scope.

        Returns:
            ast.AST: Transformed node.
        """
        old_scope, self.ctx.scope =  self.ctx.scope, self.ctx.scope.new_child(node)
        self.generic_visit(node)
        self.ctx.scope = old_scope
        return node

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        """Process an if statement with possible unpack optimization.

        If the node matches a special unpacking pattern, its body is
        returned directly. This mechanism is also used to temporarily
        group a set of statements into a single block.

        Args:
            node (ast.If): If statement node.

        Returns:
            ast.If | list[ast.stmt]: Transformed node or its body.
        """
        if is_unpack_if(node):
            return node.body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process a function definition within a new scope."""
        return self.visit_new_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Process an async function definition within a new scope."""
        return self.visit_new_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Process a class definition within a new scope."""
        return self.visit_new_scope(node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        """Process a list comprehension within a new scope."""
        return self.visit_new_scope(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        """Process a set comprehension within a new scope."""
        return self.visit_new_scope(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        """Process a dictionary comprehension within a new scope."""
        return self.visit_new_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        """Process a generator expression within a new scope."""
        return self.visit_new_scope(node)

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        """Process a with statement containing a directive.

        If the node is associated with a parsed directive, the corresponding
        construct is applied to its body and replaces the original node.

        Args:
            node (ast.With): With statement node.

        Returns:
            ast.With | list[ast.stmt]: Transformed node or replacement body.
        """
        if directive := self.ctx.directives.get(node):
            node.body = construct(directive.construct, node.body, self.ctx)
            self.generic_visit(node)
            return node.body
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | list[ast.stmt]:
        """Process a directive used as a standalone statement.

        The directive is transformed into a try block to ensure proper
        execution semantics of the resulting statements.

        Args:
            node (ast.Expr): Expression node.

        Returns:
            ast.Expr | list[ast.stmt]: Transformed node or replacement body.
        """
        if directive := self.ctx.directives.get(node):
            tmp = ast.Try(construct(directive.construct, [], self.ctx))
            self.generic_visit(tmp)
            return tmp.body
        self.generic_visit(node)
        return node
