import shutil
import sys
import types
import ast
import os
import typing
import hashlib
import platform
import py_compile
import tempfile

import importlib.util
import importlib.machinery
import importlib.metadata

from omp4py.core.processor.nodes import ParserArgs

__all__ = ['build', 'search_cache', 'get_cache_dir', 'gen_cache_key', '__version__']

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

try:
    import Cython.Build.Inline as cython_inline
    import Cython.Compiler.Options as cython_options
    import cython


    def check_cython():
        pass
except ImportError as ex:
    exstr = str(ex)


    def check_cython():
        raise RuntimeError("'cython' and 'setuptools' are required to compile:", exstr)


def gen_cache_key(code: str, _compile: bool, compiler_args: dict):
    value: str
    if _compile:
        check_cython()
        value = str((code, _compile, compiler_args, __version__, sys.version_info, cython_inline.Cython.__version__))
    else:
        value = str((code, _compile, compiler_args, __version__))

    return '__omp4py__' + hashlib.sha256(value.encode('utf-8')).hexdigest()


def get_cache_dir():
    if "OMP4PY_CACHE_DIR" in os.environ:
        return os.environ["OMP4PY_CACHE_DIR"]

    parent: str | None = None
    system: str = platform.system()
    if system == "Windows":
        parent = os.getenv('TMP')
    elif system == "Darwin":
        parent = os.path.expanduser("~/Library/Caches")
    elif system == "Linux":
        parent = os.path.expanduser("~/.cache")

    if parent and os.path.isdir(parent):
        return os.path.join(parent, "omp4py")

    # last fallback
    return os.path.expanduser("~/.omp4py")


def env(module: types.ModuleType):
    import omp4py.runtime as __pure_omp
    from omp4py import _runtime as __omp
    module.__dict__['__ompp'] = __pure_omp
    module.__dict__['__omp'] = __omp


def load_dynamic(module: types.ModuleType, name: str, path: str) -> typing.Any:
    env(module)
    spec = importlib.util.spec_from_file_location(name, location=path)
    new_module = importlib.util.module_from_spec(spec)
    new_module.__dict__.update({a: b for a, b in module.__dict__.items() if a not in new_module.__dict__})
    spec.loader.exec_module(new_module)
    fc: typing.Any = new_module.__dict__[new_module.__dict__['__omp4py__']]

    if '__omp4py_modules' in new_module.__dict__:
        name: str
        for name in new_module.__dict__['__omp4py_modules']:
            new_module.__dict__[name] = module.__dict__[name]
        if '__omp4py_globals' in new_module.__dict__:
            new_module.__dict__['__omp4py_globals'] = module.__dict__
    else:
        if hasattr(fc, '__globals__'):
            fc.__globals__ = module.__dict__
        else:
            for f in dir(fc):
                if hasattr(getattr(fc, f), '__globals__'):
                    f.__globals__ = module.__dict__

    return fc


_fast_cache: dict[str, set[str]] = {}


def search_cache(module: types.ModuleType, cache_dir: str, cache_key: str) -> typing.Any:
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    if cache_dir not in _fast_cache:
        if not os.path.isdir(cache_dir):
            return None
        _fast_cache[cache_dir] = set(os.listdir(cache_dir))

    for ext in importlib.machinery.SOURCE_SUFFIXES + importlib.machinery.EXTENSION_SUFFIXES:
        if cache_key + ext in _fast_cache[cache_dir]:
            return load_dynamic(module, cache_key, os.path.join(cache_dir, os.path.join(cache_key) + ext))


def build(fc: typing.Any, name: str, module: types.ModuleType, omp_ast: ast.Module, ann: list[ast.expr],
          cache_key: str, args: ParserArgs) -> typing.Any:
    if args.cache or args.compile:
        os.makedirs(args.cache_dir, exist_ok=True)
        omp_ast.body.append(ast.Assign(targets=[ast.Name(id='__omp4py__', ctx=ast.Store())], value=ast.Constant(name)))
        ast.fix_missing_locations(omp_ast.body[-1])

    if not args.compile:
        if args.cache:
            py_file: str = os.path.join(args.cache_dir, cache_key) + '.py'
            with open(py_file, "w") as f:
                f.write(ast.unparse(omp_ast))
            py_compile.compile(py_file)

        omp_object = compile(omp_ast, filename=module.__file__, mode="exec")
        result: dict[str, typing.Any] = {}
        exec(omp_object, module.__dict__, result)
        env(module)

        return result[name]

    check_cython()
    node: ast.AST = omp_ast.body[0]
    while isinstance(node, ast.If):
        node = node.body[0]

    with tempfile.TemporaryDirectory(prefix='omp4py') as build_dir:
        src_file: str = os.path.join(build_dir, cache_key) + '.py'
        define_macros = []
        c_include_dirs = []
        with open(src_file, "w") as f:
            _resolve_imports(args, f, module, fc, ann, define_macros, c_include_dirs)
            f.write(f'__omp4py__="{name}"\n')
            f.write(ast.unparse(node))

        extension = cython_inline.Extension(
            name=cache_key,
            sources=[src_file],
            include_dirs=c_include_dirs or None,
            define_macros=define_macros or None,
        )
        compiler_args = args.compiler_args.copy()
        compiler_args['freethreading_compatible'] = True

        build_extension = cython_inline._get_build_extension()
        build_extension.extensions = cython_inline.cythonize(
            [extension],
            compiler_directives=compiler_args,
            quiet=not args.debug,
            language_level="3",
            annotate=args.debug, )
        build_extension.build_temp = os.path.dirname(src_file)
        build_extension.build_lib = args.cache_dir
        build_extension.run()

        if args.debug:
            shutil.copy(src_file[:-2] + 'html',
                        os.path.join(os.getcwd(), fc.__qualname__ + os.path.basename(src_file[:-2] + 'html')))

    env(module)
    return load_dynamic(module, cache_key,
                        os.path.join(args.cache_dir, cache_key) + build_extension.get_ext_filename(''))


def _resolve_imports(args: ParserArgs, f: typing.TextIO, module: types.ModuleType, fc: typing.Any,
                     ann: list[ast.expr], define_macros: list[str], c_include_dirs: list[str]) -> None:
    symbols: set[str]
    if hasattr(fc, '__code__'):
        symbols = set(fc.__code__.co_names)
    else:
        symbols = set(sum([getattr(fc, f).__code__.co_names for f in dir(fc)
                           if hasattr(getattr(fc, f), '__code__')], []))

    if len(ann) > 0:  # add annotations dependencies
        value: ast.expr
        node: ast.AST
        for value in ann:
            for node in ast.walk(value):
                if isinstance(node, ast.Name):
                    symbols.add(node.id)

    shadow_globals: bool = 'globals' in symbols
    omp4py: types.ModuleType = sys.modules['omp4py']
    copy_imports: set[types.ModuleType] = {omp4py, cython}
    symbols -= set(__builtins__.keys())
    symbols &= set(module.__dict__.keys())

    cimport: str = ''
    if not args.pure:
        cimport = 'cython.cimports.'
        f.write(f'import {cimport}omp4py.cruntime as __omp\n')

        if gen_native_types(ann):
            f.write(f'from {cimport}omp4py.cruntime.basics.compilertypes import *\n')

    else:
        f.write('import omp4py.runtime as __ompp\n')

    if 'numpy' in sys.modules:
        import numpy
        copy_imports.add(numpy)
        c_include_dirs.append(numpy.get_include())
        define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

        if 'np_pythran' in sys.modules:
            import pythran
            c_include_dirs.append(pythran.get_include())

    name: str
    for name in sorted(symbols):
        import_: typing.Any = module.__dict__[name]
        iname: str = import_.__name__ if hasattr(import_, '__name__') else name

        if import_ in copy_imports:
            prefix: str = ''
            if iname == 'omp4py':
                prefix = cimport
                symbols.remove(name)
            f.write(f'import {prefix}{iname} as {name}\n')
        elif hasattr(import_, '__module__') and import_.__module__ in copy_imports:
            f.write(f'from {import_.__module__} import {name}\n')
        elif name in omp4py.__dict__ and import_ == omp4py.__dict__[name] and 'runtime' in import_.__module__:
            f.write(f'from {cimport}omp4py import {name}\n')
            symbols.remove(name)

    f.write(f'__omp4py_modules = {sorted(symbols)}\n')
    for name in sorted(symbols):
        f.write(f'{name} = ...\n')

    if shadow_globals:
        f.write(f'__omp4py_globals = ...\nglobals = lambda: __omp4py_globals\n')


def gen_native_types(ann: list[ast.expr]) -> bool:
    found: bool = True
    expr: ast.Expr
    for expr in ann:
        if isinstance(expr, ast.Name) and expr.id in ('int', 'float'):
            expr.id = f'omp_{expr.id}'
            found = True
    return found
