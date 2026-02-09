"""TODO: write the docstring."""
import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

from omp4py.core.imports.loader import FOMP as __OMP__
from omp4py.core.preprocessor import obj2ast
from omp4py.core.preprocessor.transformers import OmpTransformer
from omp4py.core.options import Options

__all__ = ["process_file", "process_object", "process_source"]


def process_object[T: Callable[..., Any] | type](arg: T, opt: Options) -> T:
    filename: str
    data: str
    module: ast.Module
    filename, data, module = obj2ast.from_object(arg)
    module: ast.Module = process(module, False, data, filename, opt)
    globals: ModuleType = sys.modules[arg.__module__]

    # TODO: use importlib to use pyc cache
    code: CodeType = compile(source=module, filename=filename, mode="exec")
    result: dict[str, Any] = {}
    exec(code, globals.__dict__, result)  # noqa: S102

    import omp4py.runtime as __omp  # noqa: PLC0415

    globals.__dict__["__omp"] = __omp

    return result[arg.__name__]


def process_source(data: str, filename: str, opt: Options) -> ast.Module:
    return process(ast.parse(data, filename), True, data, filename, opt)


def process_file(filename: str, opt: Options) -> str:
    with open(filename) as f:
        data: str = f.read()
    module: ast.Module = process_source(data, filename, opt)

    target: Path = Path(filename).parent / __OMP__ / Path(filename).name
    if not target.parent.exists():
        target.parent.mkdir(exist_ok=True)
    with target.open("w") as f:
        f.write(ast.unparse(module))

    return filename


def process(module: ast.Module, is_module: bool, full_source: str, filename: str, opt: Options) -> ast.Module:
    transformer: OmpTransformer = OmpTransformer(full_source, filename, module, is_module, opt)

    return transformer.transform()
