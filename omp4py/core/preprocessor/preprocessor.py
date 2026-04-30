"""OpenMP preprocessing.

This module provides the high-level entry points for applying the OpenMP-like
transformation system to Python code.

It acts as the central orchestration layer that connects:
- Source extraction utilities (`obj2ast`)
- AST-based transformation engine (`OmpTransformer`)
- Runtime code generation

The preprocessor supports three input modes:
- Python source files
- Raw source code strings
- Python objects (functions or classes)

All inputs are normalized into an AST representation and passed to the
transformation pipeline, where OpenMP-like constructs are rewritten into
standard Python AST with runtime support calls.

The transformed code can either be executed directly (object mode) or
written back to disk (file mode).
"""

from __future__ import annotations

import ast
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from types import CodeType, ModuleType

from omp4py.core.imports.loader import FOMP as __OMP__
from omp4py.core.modifiers.engine import ModifierEngine
from omp4py.core.preprocessor import obj2ast
from omp4py.core.preprocessor.transformers import OmpTransformer

if typing.TYPE_CHECKING:
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
    filename: str
    data: str
    module: ast.Module
    filename, data, module = obj2ast.from_object(arg)
    module: ast.Module = process(module, data, filename, opt)
    module_globals: ModuleType = sys.modules[arg.__module__]

    # TODO: use importlib to use pyc cache
    code: CodeType = compile(source=module, filename=filename, mode="exec")
    result: dict[str, typing.Any] = {}
    exec(code, module_globals.__dict__, result)  # noqa: S102

    import omp4py.runtime as _omp  # noqa: PLC0415

    module_globals.__dict__["_omp"] = _omp

    return result[arg.__name__]


def process_source(data: str, filename: str, opt: Options) -> ast.Module:
    """Preprocess Python source code and return the transformed AST.

    The source code is parsed into an abstract syntax tree and passed
    through the OpenMP transformation pipeline.

    Args:
        data (str): Python source code.
        filename (str): Virtual or real filename associated with the source.
        opt (Options): Preprocessing options.

    Returns:
        ast.Module: Transformed module AST.
    """
    return process(ast.parse(data, filename), data, filename, opt)


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
    with open(filename) as f:
        data: str = f.read()
    module: ast.Module = process_source(data, filename, opt)

    target: Path = Path(filename).parent / __OMP__ / Path(filename).name
    if not target.parent.exists():
        target.parent.mkdir(exist_ok=True)
    with target.open("w") as f:
        f.write(ast.unparse(module))

    return str(target)


def process(module: ast.Module, full_source: str, filename: str, opt: Options) -> ast.Module:
    """Apply the OpenMP transformation pipeline to an AST module.

    This is the internal entry point used by all preprocessing modes. It
    creates an `OmpTransformer`, applies all AST rewrites, and optionally
    writes the transformed source to a debug dump file.

    Args:
        module (ast.Module): Input module AST.
        full_source (str): Original source code.
        filename (str): Source filename.
        opt (Options): Preprocessing options.

    Returns:
        ast.Module: Transformed module AST.
    """
    transformer: OmpTransformer = OmpTransformer(full_source, filename, module, opt)
    modifier_engine = ModifierEngine(opt)

    modifier_engine.stage(module)
    new_module = transformer.transform()
    modifier_engine.stage(module, "omp_transformer")

    if opt.dump is not None:
        with open(opt.dump, "w") as f:
            f.write(ast.unparse(new_module))
    return new_module
