"""TODO: write docstring."""

from hashlib import new
import ast
import typing
from collections.abc import Callable

from omp4py.core.parser import syntax_error, parse
from omp4py.core.parser.tree import Directive, Construct, Parallel, Span
from omp4py.core.preprocessor.transformers import parallelism
from omp4py.core.preprocessor.transformers.context import Context, Params, SymbolTable, global_symtable

__all__ = ["OmpTransformer", "Params"]


constructs: dict[type[Construct], Callable[[Construct, list[ast.stmt], Context], list[ast.stmt]]] = {
    Parallel: parallelism.parallel,
}


def not_implemented(ctr: Construct, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:
    msg: str = f"{ctr.name} not implemented"
    raise syntax_error(msg, ctr.span, ctx.full_source, ctx.filename)


def get_construct(d: Directive) -> Callable[[Construct, list[ast.stmt], Context], list[ast.stmt]]:
    return constructs.get(d.construct.__class__, not_implemented)


def take_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ctx: Context) -> ast.expr | None:
    i: int
    child: ast.expr
    name: str
    for i, child in enumerate(node.decorator_list):
        if isinstance(child, ast.Name):
            name = child.id
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            name = child.func.id
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            name = child.func.attr
        elif isinstance(child, ast.Attribute):
            name = child.attr
        else:
            continue
        if name == ctx.params.alias:
            if ctx.decorator is not None:
                pass  # TODO: raise multiple omp decorator error or module with multiple omp
            ctx.decorator = node.decorator_list.pop(i)
            break


class OmpTransformer(ast.NodeTransformer):
    ctx: Context

    def __init__(self, full_source: str, filename: str, module: ast.Module, is_module: bool, params: Params):
        self.ctx = Context(full_source, filename, module, is_module, params)

    def transform(self) -> ast.Module:
        if len(self.ctx.node_stack) > 0:
            self.ctx.node_stack.pop()
            self.visit(self.ctx.module)
        return self.ctx.module

    def visit(self, node: ast.AST) -> ast.AST:
        self.ctx.node_stack.append(node)
        new_node: ast.AST = super().visit(node)
        self.ctx.node_stack.pop()
        return new_node

    def multiple_visit(self, nodes: list[ast.AST]) -> list[ast.AST]:
        new_values: list[ast.AST] = []
        node: ast.AST
        for node in nodes:
            value: ast.AST = self.visit(node)
            if value is None:
                continue
            if isinstance(value, ast.AST):
                new_values.append(value)
                continue
            new_values.append(value)
        nodes[:] = new_values
        return new_values

    def visit_Module(self, node: ast.Module) -> ast.Module:
        if not self.ctx.is_module:
            self.ctx.symtable = global_symtable(self.ctx.full_source, self.ctx.filename)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        take_decorator(node, self.ctx)
        parent_symtable: SymbolTable = self.ctx.symtable
        self.ctx.symtable = parent_symtable.new_child()
        self.generic_visit(node)
        self.ctx.symtable = parent_symtable
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        take_decorator(node, self.ctx)
        parent_symtable: SymbolTable = self.ctx.symtable
        self.ctx.symtable = parent_symtable.new_child()
        self.generic_visit(node)
        self.ctx.symtable = parent_symtable
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        take_decorator(node, self.ctx)
        parent_symtable: SymbolTable = self.ctx.symtable
        self.ctx.symtable = parent_symtable.new_child()
        self.generic_visit(node)
        self.ctx.symtable = parent_symtable
        return node

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        self.multiple_visit(typing.cast("list[ast.AST]", node.items))
        if self.ctx.directive is not None:
            directive: Directive = self.ctx.directive
            self.ctx.directive = None
            node.body = get_construct(directive)(directive.construct, node.body, self.ctx)
            self.multiple_visit(typing.cast("list[ast.AST]", node.body))
            return node.body
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | list[ast.stmt]:
        self.generic_visit(node)
        if self.ctx.directive is not None:
            directive: Directive = self.ctx.directive
            self.ctx.directive = None
            return get_construct(directive)(directive.construct, [], self.ctx)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        name: str = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name != self.ctx.params.alias:
            return node

        if len(node.args) != 1:
            msg: str = f"{name} () takes exactly one argument ({len(node.args)} given)"
            raise syntax_error(msg, Span.from_ast(node), self.ctx.full_source, self.ctx.filename)

        expr: ast.expr = node.args[0]
        if not isinstance(expr, ast.Constant) or not isinstance(expr.value, str):
            msg: str = f"{name} () argument needs constant string expression"
            raise syntax_error(msg, Span.from_ast(node), self.ctx.full_source, self.ctx.filename)

        if isinstance(self.ctx.node_stack[-2], ast.Expr):
            return node

        if isinstance(self.ctx.node_stack[-3], ast.With):
            with_node: ast.With = typing.cast("ast.With", self.ctx.node_stack[-3])
            withitem_node: ast.withitem = typing.cast("ast.withitem", self.ctx.node_stack[-2])

            if len(with_node.items) > 1:
                msg: str = "cannot use more than one directive item in a 'with' statement"
                raise syntax_error(msg, Span.from_ast(node), self.ctx.full_source, self.ctx.filename)
            if withitem_node.optional_vars is not None:
                msg: str = "'as' is not allowed in a directive item"
                raise syntax_error(
                    msg, Span.from_ast(withitem_node.optional_vars), self.ctx.full_source, self.ctx.filename
                )

        directive: str = typing.cast("str", ast.get_source_segment(self.ctx.full_source, expr))

        self.ctx.directive = parse(directive, expr.lineno, expr.col_offset)

        return node
