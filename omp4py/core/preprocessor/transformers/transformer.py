"""TODO: write docstring."""

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
    return syntax_error(message, span, ctx.full_source, ctx.filename)


@singledispatch
def construct(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    msg: str = f"'{ctr.name}' not implemented"
    raise syntax_error_ctx(msg, ctr.span, ctx)


def find_decorator(ctx: Context, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
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
    ctx: Context

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def visit(self, node: ast.AST) -> ast.AST:
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        find_decorator(self.ctx, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
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
    ctx: Context

    def __init__(self, full_source: str, filename: str, module: ast.Module, is_module: bool, opt: Options) -> None:
        self.ctx = Context(full_source, filename, module, is_module, opt)

    def transform(self) -> ast.Module:
        if len(self.ctx.node_stack) > 0:
            self.ctx.node_stack.pop()
            OmpParser(self.ctx).visit(self.ctx.module)
            self.visit(self.ctx.module)
        return self.ctx.module

    def visit(self, node: ast.AST) -> ast.AST:
        self.ctx.symtable.update(node)
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def visit_new_scope[T: ast.AST](self, node: T) -> T:
        old_symtable, self.ctx.symtable = self.ctx.symtable, self.ctx.symtable.new_child()
        self.generic_visit(node)
        self.ctx.symtable = old_symtable
        return node

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        if is_unpack_if(node):
            return node.body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self.visit_new_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self.visit_new_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self.visit_new_scope(node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        return self.visit_new_scope(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        return self.visit_new_scope(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        return self.visit_new_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        return self.visit_new_scope(node)

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        if directive := self.ctx.directives.get(node):
            node.body = construct(directive.construct, node.body, self.ctx)
            self.generic_visit(node)
            return node.body
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | list[ast.stmt]:
        if directive := self.ctx.directives.get(node):
            tmp = ast.Try(construct(directive.construct, [], self.ctx))
            self.generic_visit(tmp)
            return tmp.body
        self.generic_visit(node)
        return node
