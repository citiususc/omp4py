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


    def check_cython():
        pass
except ImportError as ex:
    def check_cython():
        raise RuntimeError("compile error: cython and setuptools are required to compile") from ex


def gen_cache_key(code: str, _compile: bool, compiler_args: dict):
    value: str
    if _compile:
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


def load_dynamic(name: str, path: str) -> typing.Any:
    spec = importlib.util.spec_from_file_location(name, location=path)
    new_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_module)

    return new_module.__dict__[new_module.__dict__['__omp4py__']]


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
            return load_dynamic(cache_key, os.path.join(cache_dir, os.path.join(cache_key) + ext))


def build(name: str, module: types.ModuleType, omp_ast: ast.Module, cache_key: str, args: ParserArgs) -> typing.Any:
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
        pyx_file: str = os.path.join(build_dir, cache_key) + '.pyx'
        with open(pyx_file, "w") as f:
            f.write('cimport omp4py.cruntime as __omp\n')
            f.write('import cython\n')
            f.write(f'__omp4py__="{name}"\n')
            f.write(ast.unparse(node))

        define_macros = []
        c_include_dirs = []

        if 'numpy' in sys.modules:
            import numpy
            c_include_dirs.append(numpy.get_include())
            define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

        extension = cython_inline.Extension(
            name=cache_key,
            sources=[pyx_file],
            include_dirs=c_include_dirs or None,
            define_macros=define_macros or None,
        )
        compiler_args = args.compiler_args.copy()
        compiler_args['freethreading_compatible'] = True

        build_extension = cython_inline._get_build_extension()
        build_extension.extensions = cython_inline.cythonize(
            [extension],
            include_path=['.', os.path.join(args.cache_dir, 'include')],
            compiler_directives=compiler_args,
            quiet=not args.debug)
        build_extension.build_temp = os.path.dirname(pyx_file)
        build_extension.build_lib = args.cache_dir
        build_extension.run()

    env(module)
    return load_dynamic(cache_key, os.path.join(args.cache_dir, cache_key) + build_extension.get_ext_filename(''))
