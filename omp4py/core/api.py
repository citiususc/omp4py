"""OpenMP-aware Python preprocessing utilities.

This module provides the `omp` function and related types for specifying
OpenMP directives in Python code and for preprocessing Python callables,
classes, modules, or files to automatically generate parallelized code.

Key components:

- `OMPArgs` (`TypedDict`): Defines optional keyword arguments used as metadata
  or aliases for `omp` directives and decorators.

- `_OMPFuncType` (`Protocol`): Protocol describing the full set of `omp`
  function signatures. Use this type for static type checking when storing
  references to `omp`.

- `omp` (overloaded function):
    1. **Directive context manager** (`omp(value: str, /, **kwargs)`):
       Provides a minimal no-op context manager for OpenMP directives.
    2. **Callable/class/module preprocessing** (`omp(value: T, /, **kwargs)`):
       Preprocesses Python objects containing `omp` calls into parallel code.
    3. **File-based preprocessing** (`omp(value: None, /, py: str, **kwargs)`):
       Reads a Python file, applies OpenMP transformations, and writes a new file.
    4. **Decorator factory** (`omp(value: None, /, **kwargs)`):
       Creates a new `omp` decorator with default metadata for future use.

Example usage:

from omp4py import omp

# Directive context manager
with omp("parallel for"):
    ...

# Preprocessing a function
@omp
def my_func(): ...

# Preprocessing a module (__init__.py)
omp(my_module.__spec__)
import my_module.foo

# File preprocessing
omp(py="script.py")

# Creating a decorator with defaults
my_omp = omp(alias="my_omp")
"""
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from fileinput import filename
from functools import partial
from types import ModuleType
from typing import Any, Protocol, TypedDict, Unpack, cast, overload

from omp4py.core.imports.loader import set_omp_package
from omp4py.core.options import Options

__all__ = ["OmpType", "omp"]


class OmpKwargs(TypedDict, total=False):
    """Keyword arguments accepted by the `omp` entry point.

    This type defines the high-level configuration that can be passed to
    the `omp` function. These arguments are translated into `Options`
    and control how preprocessing and transformation are applied.

    Attributes:
        alias (str):
            Name used to identify OpenMP-like directives in the source
            code (default: `"omp"`).

        compile (bool):
            Enables compilation of the transformed code after preprocessing.

        modifiers (dict[str, bool]):
            Enables or disables specific source modifiers applied before
            or after the transformation process.

        device (str):
            Default target device identifier used during transformation.

        compiler (str):
            Compiler backend used when `compile` is enabled.

        parsers (list[str]):
            Alternative parsers that extend directive syntax beyond the scope
            of the default `alias` parser.

        compiler_args (dict[str, Any]):
            Additional arguments passed to the selected compiler backend.

        modifiers_args (dict[str, Any]):
            Additional configuration values associated with source modifiers.

        args_append (bool):
            If `True`, successive calls to `omp` will merge collection-type
            arguments (`list`, `set`, `dict`) instead of replacing them.
            This allows incremental configuration across multiple calls.

        dump (str | bool):
            If set, writes the transformed source code to a file. A string
            specifies the output path, while `True` enables a default
            location.

        debug (bool):
            Enables debug mode with additional checks or logging.
    """
    alias: str
    compile: bool
    modifiers: dict[str, bool]
    device: str
    compiler: str
    parsers: list[str]
    compiler_args: dict[str, Any]
    modifiers_args: dict[str, Any]
    args_append: bool
    dump: str | bool
    debug: bool


class OmpType(Protocol):
    """Protocol defining the signatures of all `omp` overloads."""

    @overload
    def __call__(self, value: str, /, **kwargs: Unpack[OmpKwargs]) -> AbstractContextManager: ...

    @overload
    def __call__[T: Callable[..., Any] | type | ModuleType](self, value: T, /, **kwargs: Unpack[OmpKwargs]) -> T: ...

    @overload
    def __call__(self, value: None = None, /, py: str = "", **kwargs: Unpack[OmpKwargs]) -> str: ...

    @overload
    def __call__(self, value: None = None, /, **kwargs: Unpack[OmpKwargs]) -> Callable[..., Any]: ...


@overload
def omp(value: str, /, **kwargs: Unpack[OmpKwargs]) -> AbstractContextManager:
    """Define an OpenMP directive (optionally with clauses) and provide a safe Python context.

    This function allows you to specify an OpenMP directive, optionally including
    one or more clauses, in Python code. OpenMP directives are used to instruct
    the preprocessor to parallelize loops, regions, or sections of code, while
    clauses modify the behavior of these directives (e.g., specifying the number
    of threads or the sharing of variables).

    The function returns a minimal no-op context manager that acts as a placeholder,
    ensuring that Python code using `with omp(...):` remains syntactically valid
    even when the directive or clauses are not processed or removed.

    Additional options can be provided via `**kwargs` to control preprocessing behavior.

    Args:
        value (str): The OpenMP directive (with optional clauses) to apply
            (e.g., `"parallel"`, `"parallel for"`, `"critical"`,
            `"parallel for num_threads(4)"`, etc.).

        **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
            additional options for the preprocessor.
            See `OmpKwargs` for details.

    Returns:
        typing.ContextManager: A no-op context manager that exists only to prevent
        Python runtime errors when using the directive (and optional clauses)
        in a `with` block.
    """


@overload
def omp[T: Callable[..., Any] | type | ModuleType](value: T, /, **kwargs: Unpack[OmpKwargs]) -> T:
    """Invoke the OpenMP preprocessor on a callable or class, or mark a module to be preprocessed at import time.

    This function acts as the entry point to the OpenMP-aware preprocessor.
    When called on a function or class, it analyzes all contained
    calls to :func:`omp` and transforms them into parallel OpenMP-compatible
    code. Any OpenMP directives specified inside the decorated objects are
    processed, allowing loops and regions to be parallelized automatically.

    When applied to a module, the module is marked for preprocessing and will
    be transformed automatically the first time it is imported.

    At runtime, the returned object is the processed version of the input.
    The original object may be replaced with a transformed version that
    includes generated parallel code. Additional options can be provided
    via `**kwargs` to control preprocessing behavior.

    Args:
        value (Callable[..., Any] | type | ast.Module): The target object to
            preprocess. Can be a function, a class, or an AST module containing
            calls to :func:`omp`.

        **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
            additional options for the preprocessor.
            See `OmpKwargs` for details.

    Returns:
        T: The preprocessed object with OpenMP transformations applied or the module
    """


@overload
def omp(value: None = None, /, py: str = "", **kwargs: Unpack[OmpKwargs]) -> str:
    """Preprocess a Python file for OpenMP directives and generate transformed code.

    This overload of `omp` allows you to specify a Python file to be processed
    by the OpenMP-aware preprocessor. When `value` is None and `py` is provided,
    the preprocessor reads the indicated file, analyzes all calls to :func:`omp`
    inside, and generates a new Python file with the transformed parallel code.

    Additional metadata can be provided via `**kwargs` to control preprocessing
    behavior.

    Args:
        value (None, optional): Must be None or omitted for file-based preprocessing.
        py (str): Path to the Python file to preprocess. The file will be read,
            transformed according to OpenMP directives, and written back or to
            a new output file depending on preprocessor settings.

        **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
            additional metadata or directives for the preprocessor.
            See `OmpKwargs` for details.

    Returns:
        str: Path of the preprocessed Python file, after OpenMP
        transformations have been applied.
    """


@overload
def omp(value: None = None, /, **kwargs: Unpack[OmpKwargs]) -> OmpType:
    """Create a new `omp` decorator with default options for future use.

    When no positional argument is provided, this overload returns a new
    `omp` decorator function where the provided keyword arguments (`**kwargs`)
    are stored as default metadata. These defaults will automatically apply
    whenever the returned decorator is used on a function, class, or module.

    If the decorator is assigned to a name other than `omp`, the `alias`
    keyword argument should be set to that name so that the preprocessor
    can correctly detect and process it.

    Args:
        value (None, optional): Must be None or omitted for this overload.
        **kwargs (typing.Unpack[OmpKwargs]): Default keyword arguments to store
            in the new decorator. See `OmpKwargs` for details.

    Returns:
        OmpType: A new `omp` decorator function with the specified
        defaults applied.
    """


def omp(value: Any = None, /, py: str | None = None, __ka: OmpKwargs | None = None, **kwargs: Unpack[OmpKwargs]) -> Any:
    """Unified OpenMP preprocessing entry point.

    This function implements all overloads of `omp`:

    - **Directive context manager**
      (`omp(value: str, /, **kwargs)`):
      Returns a no-op context manager for an OpenMP directive.

    - **Preprocessing of callables, classes, or modules**
      (`omp[T: Callable[..., Any] | type | ModuleType](value: T, /, **kwargs)`):
      Analyzes and transforms contained `omp` calls into parallel code.

    - **File-based preprocessing**
      (`omp(value: None, /, py: str, **kwargs)`):
      Reads a Python file, applies OpenMP transformations, and writes a new file.

    - **Decorator factory**
      (`omp(value: None, /, **kwargs)`):
      Returns a new `omp` decorator with default metadata for future use.
      Use the `alias` keyword if assigning a different name for detection.

    Additional metadata or directives can be provided via `**kwargs` (see `OmpKwargs`).

    Args:
        value (Any, optional): Determines the overload behavior.
        py (str | None, optional): Path to a Python file for file-based preprocessing.
        __ka (OmpKwargs | None): Previous **kwargs used for argument appending.
        **kwargs (Unpack[OmpKwargs]): Metadata or default directives for the preprocessor.

    Returns:
        Any: The return type depends on the overload:
            - `AbstractContextManager` for directives
            - The processed callable/class/module
            - `str` for file-based preprocessing
            - `OmpType` a new `omp` for a decorator factory
    """
    if __ka is not None:
        for key, arg in __ka.items():
            match arg:
                case list():
                    kwargs[key] = arg + kwargs[key]
                case set() | dict():
                    kwargs[key] = arg | kwargs[key]
        __ka = None
    if kwargs.pop("args_append", False):
        __ka = deepcopy(cast(OmpKwargs,kwargs))

    if py is None:
        if value is None:
            return partial(omp, py=py, __ka=__ka, **kwargs)
        elif isinstance(value, str):
            return contextmanager(lambda: (yield))()
        elif isinstance(value, ModuleType):
            return set_omp_package(value, Options(is_module=True, **kwargs))
        from omp4py.core.preprocessor import process_object  # noqa: PLC0415 Lazy import, only when needed

        return process_object(value, Options(**kwargs))
    else:
        from omp4py.core.preprocessor import process_file  # noqa: PLC0415 Lazy import, only when needed

        return process_file(py, Options(is_module=True, **kwargs))
