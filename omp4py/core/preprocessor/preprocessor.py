"""OpenMP preprocessing.

This module provides the high-level entry points for applying the
OpenMP-like transformation system to Python code.

It acts as the central orchestration layer that connects:
- Source extraction utilities (`obj2ast`)
- AST modifier stages (`ModifierEngine`)
- AST-based transformation engine (`OmpTransformer`)
- Runtime and optional compiled code generation

The preprocessor supports multiple input modes:
- Python source files
- Raw source code strings
- Python objects (functions or classes)

All inputs are normalized into an AST representation and passed through
the preprocessing pipeline.

The transformation pipeline is divided into multiple stages:

1. Pre-transformation modifiers:
   Optional AST modifiers may run before OpenMP transformation to adapt,
   normalize, optimize, or extend the source tree.

2. OpenMP transformation:
   The `OmpTransformer` rewrites OpenMP-like constructs into standard
   Python AST nodes combined with runtime support calls.

3. Post-transformation modifiers:
   Additional modifiers may run after the main transformation stage to
   apply backend-specific optimizations, typing rewrites, or compilation
   adjustments.

The resulting transformed code can then be:
- Executed directly in pure Python mode
- Cached as generated Python source
- Compiled into native extensions using Cython

The modifier system is fully configurable through preprocessing options
and allows custom transformation passes to be integrated into the
pipeline.
"""

from __future__ import annotations

import ast
import hashlib
import py_compile
import sys
import typing
from collections.abc import Callable
from importlib.machinery import ModuleSpec, PathFinder, SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from omp4py.core.imports.loader import OMP_FOLDER
from omp4py.core.modifiers.engine import ModifierEngine
from omp4py.core.preprocessor import obj2ast
from omp4py.core.preprocessor.cbuild import cythonize
from omp4py.core.preprocessor.transformers import OmpTransformer

if typing.TYPE_CHECKING:
    from types import CodeType, ModuleType

    from omp4py.core.options import Options

__all__ = ["process_file", "process_object", "process_source"]


def process_object[T: Callable[..., typing.Any] | type](arg: T, opt: Options) -> T:
    """Preprocess a Python callable or class and return the transformed object.

    The input object is converted into source code and an AST, processed by
    the OpenMP transformation pipeline, compiled, and executed in the
    original module namespace. The transformed definition replaces the
    original one in the generated result.

    Args:
        arg (T): Function or class to preprocess.
        opt (Options): Preprocessing options.

    Returns:
        T: The transformed function or class.
    """
    opt.filename, data, module = obj2ast.from_object(arg)
    module_globals: ModuleType = sys.modules[arg.__module__]
    result: dict[str, typing.Any] = {}

    path_hash: str = hashlib.sha256(bytes(Path(opt.filename).resolve())).hexdigest()[:12]
    fullname: str = f"p{path_hash}_{arg.__module__}_{arg.__name__}"
    cache_dir = Path(opt.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if opt.compile:
        spec = PathFinder.find_spec(fullname, [opt.cache])

        if spec is None or spec.cached or opt.ignore_cache:
            result_module: ast.Module = process(module, data, opt)
            module.body.insert(
                0, ast.ImportFrom("cython.cimports.omp4py", names=[ast.alias("runtime", "_omp")], level=0),
            )
            ast.fix_missing_locations(module.body[0])
            cythonize(fullname, str(cache_dir), result_module, opt)

        spec = typing.cast("ModuleSpec", PathFinder.find_spec(fullname, [opt.cache]))

        ext_module = module_from_spec(spec)
        ext_module.__dict__.update(module_globals.__dict__)
        if spec.loader:
            spec.loader.exec_module(ext_module)

        return ext_module.__dict__[arg.__name__]

    pyfile = (cache_dir / fullname).with_suffix(".py")
    spec = typing.cast("ModuleSpec", spec_from_file_location(fullname, pyfile))

    result_code = ""
    if not pyfile.exists() or opt.ignore_cache:
        result_module: ast.Module = process(module, data, opt)
        try:
            pyfile.write_text(result_code := ast.unparse(result_module))
            py_compile.compile(str(pyfile))
        except Exception:  # noqa: BLE001, S110
            pass

    codecache: CodeType | str | None = None
    if spec and spec.cached and Path(spec.cached).exists():
        codecache = SourcelessFileLoader("f", spec.cached).get_code("f")
    if codecache is None:
        codecache = result_code or Path(pyfile).read_text()

    exec(codecache, module_globals.__dict__, result)  # noqa: S102
    module_globals.__dict__["_omp"] = sys.modules["omp4py.runtime"]

    return result[arg.__name__]


def process_source(data: str, filename: str, opt: Options) -> str:
    """Preprocess Python source code and return the transformed result code.

    The source code is parsed into an abstract syntax tree and passed
    through the OpenMP transformation pipeline.

    Args:
        data (str): Python source code.
        filename (str): Filename associated with the source to store
                        the result if path is writable.
        opt (Options): Preprocessing options.

    Returns:
        ast.Module: Transformed module source code.
    """
    target = Path(filename)
    source: Path = target
    if source.parent.name == OMP_FOLDER:
        source = source.parent.parent / source.name

    opt.filename = str(source)
    module: ast.Module = process(ast.parse(data, opt.filename), data, opt)
    module.body.insert(0, ast.ImportFrom("omp4py", names=[ast.alias("runtime", "_omp")], level=0))
    ast.fix_missing_locations(module.body[0])
    result_code = ast.unparse(module)

    try:
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
        with target.open("w") as f:
            f.write(result_code)
    except Exception:  # noqa: BLE001, S110
        pass

    return result_code


def process_file(filename: str, opt: Options) -> str:
    """Preprocess a Python source file and write the transformed code.

    The input file is read, transformed, and written into the generated
    `__omp__` directory using the same filename.

    Args:
        filename (str): Path to the Python file to preprocess.
        opt (Options): Preprocessing options.

    Returns:
        str: Path to the generated transformed file.
    """
    omp_path = Path(filename).parent / OMP_FOLDER / Path(filename).name  # same as loader
    process_source(omp_path.read_text(), str(omp_path), opt)
    return str(omp_path)


def process(module: ast.Module, full_source: str, opt: Options) -> ast.Module:
    """Apply the OpenMP transformation pipeline to an AST module.

    This is the internal entry point used by all preprocessing modes. It
    creates an `OmpTransformer`, applies all AST rewrites, and optionally
    writes the transformed source to a debug dump file.

    Args:
        module (ast.Module): Input module AST.
        full_source (str): Original source code.
        opt (Options): Preprocessing options.

    Returns:
        ast.Module: Transformed module AST.
    """
    transformer: OmpTransformer = OmpTransformer(full_source, module, opt)
    modifier_engine = ModifierEngine(opt)

    modifier_engine.stage(module)
    new_module = transformer.transform()
    modifier_engine.stage(module, "omp_transformer")

    if opt.dump:
        with open(opt.dump, "w") as f:
            f.write(ast.unparse(new_module))
    return new_module
