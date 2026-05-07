"""OpenMP-aware Python preprocessing utilities.

This module provides the `omp` function and related types for specifying
OpenMP directives in Python code and for preprocessing Python callables,
classes, modules, or files to automatically generate parallelized code.

Example usage:

from omp4py import omp

# Directive context manager
with omp("parallel for"):
    ...

# Preprocessing a function
@omp
def my_func():
    ...

# Preprocessing a class with custom args
@omp(compile=True)
class my_class:
    ...

# Mark a package for preprocessing (__init__.py)
omp(pkg=__package__)
import my_module.foo

# File preprocessing
omp(py="script.py")

# Creating a new decorator with defaults
my_omp = omp(alias="my_omp", ...)
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from typing import Any, TypedDict, Unpack, overload

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

        ignore_cache (bool):
            If enabled, bypasses any existing cached artifacts and forces
            regeneration of all preprocessing and compilation outputs.
            This ensures that transformations are always performed from
            the original source, ignoring previously stored results.

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
    ignore_cache: bool
    modifiers: dict[str, bool]
    device: str
    compiler: str
    parsers: list[str]
    compiler_args: dict[str, Any]
    modifiers_args: dict[str, Any]
    args_append: bool
    dump: str | bool
    debug: bool


class OmpType:
    """Callable object representing the OpenMP preprocessing interface.

    This class encapsulates all behaviors required to interact with the
    OpenMP-aware preprocessor from Python code. Internally, instances
    store keyword arguments (`OmpKwargs`) that define preprocessing options
    and metadata. These options can be extended or overridden on each call.

    Attributes:
        __args (OmpKwargs): Stored keyword arguments used as default options
            for preprocessing. These are propagated or merged across calls.
    """

    __args: OmpKwargs

    def __init__(self, **kwargs: Unpack[OmpKwargs]) -> None:
        """Initialize a new OpenMP interface instance.

        This constructor creates a new `OmpType` object with an optional set
        of default preprocessing options. These options are stored internally
        and will be applied to all subsequent invocations of the instance,
        unless explicitly overridden.

        This mechanism enables the creation of customized decorators with
        predefined behavior, which can simplify repeated usage patterns.

        Args:
            **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments
                defining default preprocessing options and metadata.
                These values are stored internally and reused across calls.
        """
        self.__args = kwargs  # ty:ignore[invalid-assignment]

    @overload
    def __call__(self, value: str, /) -> AbstractContextManager:
        """Define an OpenMP directive and provide a safe Python context.

        This function allows you to specify an OpenMP directive, optionally including
        one or more clauses, in Python code. OpenMP directives are used to instruct
        the preprocessor to parallelize loops, regions, or sections of code, while
        clauses modify the behavior of these directives (e.g., specifying the number
        of threads or the sharing of variables).

        The function returns a minimal no-op context manager that acts as a placeholder,
        ensuring that Python code using `with omp(...):` remains syntactically valid
        even when the directive or clauses are not processed or removed.

        Args:
            value (str): The OpenMP directive (with optional clauses) to apply
                (e.g., `"parallel"`, `"parallel for"`, `"critical"`,
                `"parallel for num_threads(4)"`, etc.).

        Returns:
            typing.ContextManager: A no-op context manager that exists only to prevent
            Python runtime errors when using the directive (and optional clauses)
            in a `with` block.
        """

    @overload
    def __call__[**P, R](self, value: Callable[P, R], /, **kwargs: Unpack[OmpKwargs]) -> Callable[P, R]:
        """Invoke the OpenMP preprocessor on a callable.

        This function acts as the entry point to the OpenMP-aware preprocessor.
        When called on a function, it analyzes all contained calls to :func:`omp`
        and transforms them into parallel OpenMP-compatible code. Any OpenMP
        directives specified inside the decorated function are processed,
        allowing loops and regions to be parallelized automatically.

        The original function may be replaced with a transformed version that
        includes generated parallel code. Additional options can be provided
        via `**kwargs` to control preprocessing behavior.

        Args:
            value (Callable[P, R]): The function to preprocess. It may contain
                calls to :func:`omp` that will be transformed.

            **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
                additional options for the preprocessor.
                See `OmpKwargs` for details.

        Returns:
            Callable[P, R]: The preprocessed function with OpenMP transformations applied.
        """

    @overload
    def __call__[T](self, value: type[T], /, **kwargs: Unpack[OmpKwargs]) -> type[T]:
        """Invoke the OpenMP preprocessor on a class.

        This function acts as the entry point to the OpenMP-aware preprocessor.
        When called on a class, it analyzes all contained methods and their
        calls to :func:`omp`, transforming them into parallel OpenMP-compatible
        code. Any OpenMP directives specified inside the class methods are
        processed, allowing loops and regions to be parallelized automatically.

        The original class may be replaced with a transformed version that
        includes generated parallel code in its methods. Additional options can
        be provided via `**kwargs` to control preprocessing behavior.

        Args:
            value (type[T]): The class to preprocess. Its methods may contain
                calls to :func:`omp` that will be transformed.

            **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
                additional options for the preprocessor.
                See `OmpKwargs` for details.

        Returns:
            type[T]: The preprocessed class with OpenMP transformations applied.
        """

    @overload
    def __call__(self, *, pkg: str, **kwargs: Unpack[OmpKwargs]) -> None:
        """Mark a package for OpenMP preprocessing at import time.

        This function acts as the entry point to the OpenMP-aware preprocessor.
        When applied at package level (inside a `__init__.py` file), it marks the
        entire package for preprocessing. All modules contained in the package will
        be automatically transformed the first time they are imported.

        The transformation is performed lazily: nothing is modified immediately.
        Instead, when any module within the package is imported, the preprocessor
        analyzes its contents (functions, classes, and top-level code), processes
        any calls to :func:`omp`, and applies OpenMP-compatible transformations.

        This behavior applies globally to all modules in the package, ensuring
        consistent preprocessing across the entire package scope.

        At the package level, a caching mechanism based on Python's `__pycache__`
        system is used to avoid redundant preprocessing. Once modules in the
        package have been transformed, the results are cached and reused on
        subsequent imports when possible, improving import performance.

        Args:
            pkg (str): The `__package__` value of `__init__.py` module where the
                preprocessor will be applied. This marks all modules in the package
                for deferred preprocessing at import time.

            **kwargs (typing.Unpack[OmpKwargs]): Optional keyword arguments providing
                additional options for the preprocessor.
                See `OmpKwargs` for details.
        """

    @overload
    def __call__(self, *, py: str, **kwargs: Unpack[OmpKwargs]) -> str:
        """Preprocess a Python file for OpenMP directives and generate transformed code.

        This overload allows you to specify a Python file to be processed by the
        OpenMP-aware preprocessor. When `value` is None and `py` is provided,
        the preprocessor reads the indicated file, analyzes all calls to :func:`omp`
        inside, and generates a new Python file with the transformed parallel code.

        Additional metadata can be provided via `**kwargs` to control preprocessing
        behavior.

        Args:
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
    def __call__(self, **kwargs: Unpack[OmpKwargs]) -> OmpType:
        """Create a new decorator with default options for future use.

        This overload returns a new decorator where the provided keyword arguments
        (`**kwargs`) are stored as default metadata. These defaults will automatically
        apply whenever the returned decorator is used on a function, class, or module.

        If the decorator is assigned to a name other than `omp`, the `alias`
        keyword argument should be set to that name so that the preprocessor
        can correctly detect and process it.

        Args:
            **kwargs (typing.Unpack[OmpKwargs]): Default keyword arguments to store
                in the new decorator. See `OmpKwargs` for details.

        Returns:
            OmpType: A new decorator function with the specified defaults applied.
        """

    def __call__(
        self,
        value: Any = None,
        *,
        py: str | None = None,
        pkg: str | None = None,
        **kwargs: Unpack[OmpKwargs],
    ) -> Any:
        """Unified OpenMP preprocessing entry point.

        This function implements all overloads of `omp`:

        - **Directive context manager**
          (`omp(value: str, /)`):
          Returns a no-op context manager for an OpenMP directive.

        - **Preprocessing of callables**
          (`omp(value: Callable[P, R], /, **kwargs)`):
          Analyzes and transforms contained `omp` calls into parallel code.

        - **Preprocessing of classes**
          (`omp(value: type[T], /, **kwargs)`):
          Analyzes and transforms contained `omp` calls into parallel code.

        - **Preprocessing of modules**
          (`omp(*, pkg: str, /, **kwargs)`):
          Mark a package for OpenMP preprocessing at import time.

        - **File-based preprocessing**
          (`omp(*, py: str, **kwargs)`):
          Reads a Python file, applies OpenMP transformations, and writes a new file.

        - **Decorator factory**
          (`omp(**kwargs)`):
          Returns a new `omp` decorator with default metadata for future use.
          Use the `alias` keyword if assigning a different name for detection.

        Additional metadata or directives can be provided via `**kwargs` (see `OmpKwargs`).

        Args:
            value (Any): Determines the overload behavior.
            py (str | None): Path to a Python file for file-based preprocessing.
            pkg (str | None): Package name for deferred preprocessing overload.
            **kwargs (Unpack[OmpKwargs]): Metadata or default directives for the preprocessor.

        Returns:
            Any: The return type depends on the overload:
                - `AbstractContextManager` for directives
                - The processed callable/class/module
                - `str` for file-based preprocessing
                - `OmpType` a new `omp` for a decorator factory
        """
        if kwargs.pop("args_append", False):
            for key, arg in kwargs.items():
                match arg:
                    case list():
                        self.__args[key] = self.__args[key] + arg  # ty:ignore[invalid-key]
                    case set() | dict():
                        self.__args[key] = self.__args[key] | arg  # ty:ignore[invalid-key]
                    case _:
                        self.__args[key] = arg  # ty:ignore[invalid-key]
        else:
            self.__args.update(kwargs)  # ty:ignore[invalid-argument-type]

        if py is not None:
            from omp4py.core.preprocessor import process_file  # noqa: PLC0415 Lazy import, only when needed

            return process_file(py, Options(is_module=True, **self.__args))  # ty:ignore[unknown-argument, invalid-argument-type]

        if pkg is not None:
            from omp4py.core.imports.loader import set_omp_package  # noqa: PLC0415 Lazy import, only when needed

            set_omp_package(pkg, Options(**self.__args)) # ty:ignore[unknown-argument, invalid-argument-type]
            return None

        if value is not None:
            if isinstance(value, str):
                return contextmanager(lambda: (yield))()
            from omp4py.core.preprocessor import process_object  # noqa: PLC0415 Lazy import, only when needed

            return process_object(value, Options(**self.__args)) # ty:ignore[unknown-argument, invalid-argument-type]

        return OmpType(**self.__args)


omp = OmpType()
