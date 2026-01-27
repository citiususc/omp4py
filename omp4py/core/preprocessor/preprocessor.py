"""TODO: write the docstring."""
import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

from omp4py.core.imports.loader import _OMP as __OMP__
from omp4py.core.imports.loader import set_omp_package
from omp4py.core.preprocessor import obj2ast
from omp4py.core.preprocessor.transformers import OmpTransformer, Params

__all__ = ["Params", "process_file", "process_object", "process_source", "set_omp_package"]

defaults: Params = Params()

if defaults.pure:
    import omp4py.core.imports.pure as _


def process_object[T: Callable[..., Any] | type](arg: T, params: Params) -> T:
    filename: str
    data: str
    module: ast.Module
    filename, data, module = obj2ast.from_object(arg)
    module: ast.Module = process(module, False,data, filename, params)
    globals: ModuleType = sys.modules[arg.__module__]

    # TODO: use importlib to use pyc cache
    code: CodeType = compile(source=module, filename=filename, mode="exec")
    result: dict[str, Any] = {}
    exec(code, globals.__dict__, result)  # noqa: S102

    import omp4py.runtime as __omp  # noqa: PLC0415

    globals.__dict__["__omp"] = __omp

    return result[arg.__name__]


def process_source(data: str, filename: str, params: Params) -> ast.Module:
    return process(ast.parse(data, filename),True, data, filename, params)


def process_file(filename: str, params: Params) -> str:
    with open(filename) as f:
        data: str = f.read()
    module: ast.Module = process_source(data, filename, params)

    target: Path = Path(filename).parent / __OMP__ / Path(filename).name
    if not target.parent.exists():
        target.parent.mkdir(exist_ok=True)
    with target.open("w") as f:
        f.write(ast.unparse(module))

    return filename


def process(module: ast.Module, is_module: bool, full_source: str, filename: str, params: Params) -> ast.Module:
    transformer: OmpTransformer = OmpTransformer(full_source, filename, module, is_module, params)

    return transformer.transform()
