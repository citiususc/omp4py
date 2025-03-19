import os
import ast
import json
import typing
import inspect
import dataclasses
from typing import Any, cast
from types import ModuleType
from contextlib import contextmanager

from omp4py.core.directive import OmpDirective, OmpClause, tokenizer
from omp4py.core.processor.processor import OMP_PROCESSOR
from omp4py.core.processor.builder import build, search_cache, get_cache_dir, gen_cache_key, __version__
from omp4py.core.processor.nodes import NodeContext, directive_node, Variables, ParserArgs

__all__ = ['omp', 'omp4py_force_pure', '__version__']

T = typing.TypeVar('T')

_true_set = {'yes', 'true', 't', 'y', '1'}


def get_bool_config(key: str, default: bool) -> bool:
    if key in os.environ:
        return os.environ[key].lower() in _true_set
    return default


_dump: bool = get_bool_config('OMP4PY_DUMP', False)
_cache: bool = get_bool_config('OMP4PY_CACHE', False)
_force: bool = get_bool_config('OMP4PY_FORCE', False)
_cache_dir: str = get_cache_dir()
_debug: bool = get_bool_config('OMP4PY_DEBUG', False)
_compile: bool = get_bool_config('OMP4PY_COMPILE', False)
_pure: bool = get_bool_config('OMP4PY_PURE', False)
omp4py_force_pure: bool = get_bool_config('OMP4PY_FORCE_PURE', False)
try:
    _compiler_args: dict = json.loads(os.environ.get("OMP4PY_COMPILER_ARGS", "{}"))
except:
    _compiler_args: dict = {}
try:
    _compiler_args_default: dict = json.loads(os.environ.get("OMP4PY_COMPILER_DARGS", "{}"))
except:
    _compiler_args_default: dict = {}


@typing.overload
def omp(directive: str) -> typing.ContextManager:
    """
    Defines an OpenMP directive.

    Args:
        directive (str): A OpenMP directive.

    Returns:
        ContextManager: A minimal valid context for use with the 'with' statement.
    """
    pass


@typing.overload
def omp(f: typing.Callable) -> typing.Callable:
    """
    Decorate a function to search for OpenMP directives and generates code based on them.

    Args:
        f (Callable): A function containing OpenMP directives.

    Returns:
        Callable: A decorated function with the generated code.

    Raises:
        SyntaxError: If there is a syntax error in the code generation.
    """
    pass


@typing.overload
def omp(c: type[T]) -> type[T]:
    """
    Decorate a class to search for OpenMP directives  and generates code based on them.

    Args:
        c (type[T]): A class containing OpenMP directives.

    Returns:
        type[T]: A decorated class with the generated code.

    Raises:
        SyntaxError: If there is a syntax error in the code generation.
    """
    pass


@typing.overload
def omp(*, alias: str = ..., cache: bool = ..., dump: bool = ..., debug: bool = ..., pure: bool = ...,
        compile: bool = ..., compiler_args: dict = ..., compiler_modules: list = ..., force: bool = ...,
        cache_dir: str = ...) -> \
        typing.Callable[[typing.Callable], typing.Callable] | typing.Callable[[type], type]:
    """
    Create a `omp` decorator with custom arguments.

    The `omp` decorator with arguments.

    Args:
        alias (str): Create an alias for the `omp` function.
        cache (bool): Create a cache for the `omp` function.
        dump (bool): Save the decorated function to a file.
        debug (bool): Enable debug mode.
        pure (bool): Only use python runtime.
        compile (bool): Compile the decorated function to improve performance.
        compiler_args (dict): Arguments to pass to the compiler.
        compiler_modules (list): Compiler modules to pass to the compiler.
        force (bool): Force compilation even if the compiler has already been compiled.
        cache_dir (str): The directory to store the cache.

    Returns:
        Callable: A customized `omp` decorator.
    """
    pass


def omp(arg: Any = None, /, *, alias: str = "", cache: bool = _cache, dump: bool = _dump, debug: bool = _debug,
        pure: bool = _pure, compile: bool = _compile, compiler_args: dict = _compiler_args, compiler_modules: list = [],
        force: bool = _force, cache_dir: str = _cache_dir) -> Any:
    def wrap(arg):
        return omp_parse(arg,
                         ParserArgs(alias, cache, dump, debug, pure and not compile, compile,
                                    _compiler_args_default | compiler_args, list(compiler_modules), force, cache_dir))

    if arg is None:
        return wrap

    if isinstance(arg, str):
        return contextmanager(lambda: (yield))()

    return wrap(arg)


@dataclasses.dataclass
class WrapperException(Exception):
    wrap: Exception


def check_func(fm: Any, src_filename: str, src_start: int, src_line: str):
    if not inspect.getclosurevars(fm).nonlocals:
        return
    outer_name: str = fm.__qualname__.split('.')[0]
    raise SyntaxError(f"'{fm.__name__}' has variable dependencies, decorator must be set on '{outer_name}'",
                      (src_filename, src_start + 1, -1, src_line))


def omp_parse(fm: Any, args: ParserArgs) -> Any:
    src_lines: list[str]
    src_start: int
    src_name: str = fm.__name__
    src_full_lines, src_start = inspect.findsource(inspect.unwrap(fm))
    src_lines = inspect.getblock(src_full_lines[src_start:])
    src_code: str = ''.join(src_lines)
    src_module: ModuleType = inspect.getmodule(fm)
    src_filename: str = src_module.__file__
    cache_key: str = gen_cache_key(src_code, args.compile, args.compiler_args) if args.cache or args.compile else ''

    if not args.force and (cached := search_cache(src_module, args.cache_dir, cache_key)) is not None:
        return cached

    if inspect.isclass(fm):
        for name, value in inspect.getmembers(fm, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
            print(name)
            check_func(value, src_filename, src_start, src_lines[0])
    elif inspect.getclosurevars(fm).nonlocals:
        check_func(fm, src_filename, src_start, src_lines[0])

    i: int
    if len(src_lines[0]) - len(src_lines[0].lstrip()) > 0:
        indent_size: int = tokenizer.indent_size(src_full_lines)
        indent: int = (len(src_lines[0]) - len(src_lines[0].lstrip())) // indent_size
        src_lines = ["\n"] * src_start + src_lines
        for i in range(1, indent + 1):
            src_lines[src_start - i] = ' ' * (indent_size * (indent - i)) + 'if True:\n'
    else:
        src_lines = ["\n"] * src_start + src_lines

    module: ast.Module = ast.parse(''.join(src_lines), filename=src_filename)

    module = OmpTransformer(src_filename, src_lines, args).transform(module)
    module = ast.fix_missing_locations(module)

    if args.dump:
        if args.debug:
            with open(f"{os.path.basename(inspect.getfile(fm))}_{fm.__name__}_omp_d.ast", "w") as file:
                file.write(ast.dump(module, indent=4))
        with open(f"{os.path.basename(inspect.getfile(fm))}_{fm.__name__}_omp_d.py", "w") as file:
            file.write(ast.unparse(module))

    return build(src_name, src_module, module, cache_key, args)


class OmpTransformer(ast.NodeTransformer):
    ctx: NodeContext
    attibutes: bool

    def __init__(self, filename: str, src_lines: list[str], parser_args: ParserArgs) -> None:
        self.ctx = NodeContext(filename, src_lines, parser_args, '__ompp' if parser_args.pure else '__omp')
        self.attibutes = False

    def transform(self, node: ast.Module) -> ast.Module:
        try:
            return cast(ast.Module, self.visit(node))
        except WrapperException as ex:
            raise ex.wrap from None

    def visit(self, node: ast.AST) -> ast.AST:
        self.ctx.stack.append(node)
        directive_callback_old: typing.Any = self.ctx.directive_callback
        try:
            new_node: ast.AST = super().visit(node)
        except SyntaxError as ex:
            raise WrapperException(ex)
        self.ctx.directive_callback = directive_callback_old
        self.ctx.stack.pop()
        return new_node

    def remove_decorator(self, node: ast.FunctionDef | ast.ClassDef):
        node.decorator_list = list(filter(lambda x: not self.ctx.is_omp(x), node.decorator_list))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self.remove_decorator(node)
        self.ctx.variables.add(node.name)
        old_variables: Variables = self.ctx.variables.new_scope()
        new_node: ast.AST = self.generic_visit(node)
        self.ctx.variables = old_variables
        return new_node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.remove_decorator(node)
        self.ctx.variables.add(node.name)
        old: bool = self.attibutes
        self.attibutes = True
        new_node: ast.AST = self.generic_visit(node)
        self.attibutes = old
        return new_node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        self.ctx.variables.add(node.id)
        return node

    def visit_alias(self, node: ast.alias) -> ast.alias:
        if node.asname is not None:
            self.ctx.variables.add(node.asname)
        else:
            self.ctx.variables.add(node.name)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        self.ctx.variables.add(node.arg)
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.ctx.variables.add_multiple(node.names)
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        self.ctx.variables.gadd_multiple(node.names)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        if isinstance(node.value, ast.Call) and self.ctx.is_omp(node.value.func):
            return self.omp_call(node.value, [])
        return node

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.AST]:
        item: ast.withitem
        new_items: list[ast.withitem] = []
        for item in node.items:
            if isinstance(item.context_expr, ast.Call) and self.ctx.is_omp(item.context_expr.func):
                node.body = self.omp_call(item.context_expr, node.body)
            else:
                new_items.append(item)
        node.items = new_items

        self.generic_visit(node)
        if len(new_items) > 0:
            return node
        else:
            return node.body

    def omp_call(self, c: ast.Call, body: list[ast.stmt]) -> list[ast.stmt]:
        directive: OmpDirective = directive_node(self.ctx, c)

        if self.ctx.directive_callback is not None:
            self.ctx.directive_callback(self.ctx, directive)
        self.ctx.directive_callback = None

        if str(directive) in OMP_PROCESSOR:
            try:
                return OMP_PROCESSOR[str(directive)](body, list(directive.clauses), directive.args, self.ctx)
            except ValueError as ex:
                raise tokenizer.merge(directive.tokens).make_error(str(ex))

        name: str
        name_clauses: list[OmpClause]
        for name in str(directive).split():
            name_clauses = list(filter(lambda c: c.directive == name, directive.clauses))
            if name not in OMP_PROCESSOR:
                raise tokenizer.merge(directive.tokens).make_error(f"'{name}' directive is not supported yet")
            try:
                body = OMP_PROCESSOR[name](body, name_clauses, directive.args, self.ctx)
            except ValueError as ex:
                raise tokenizer.merge(directive.tokens).make_error(str(ex))

        self.ctx.directive = None
        return body
