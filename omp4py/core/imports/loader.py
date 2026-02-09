"""Import-time OpenMP preprocessing for registered Python packages.

This module integrates the `omp4py` preprocessor into Python's import
machinery, allowing entire packages to be transparently transformed
at import time. Any package registered via `set_omp_package` will have
all of its pure-Python modules automatically processed by the OpenMP
preprocessor before execution.

The mechanism is implemented using a custom `sys.meta_path` finder and
loader pair. When a module belonging to a registered package is imported,
its source code is parsed into an AST, transformed using the `omp4py`
preprocessor, and then compiled and executed as usual.
"""

import ast
import sys
import types
from collections.abc import Mapping, Sequence
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any

from omp4py.core.options import Options

__all__ = ["FOMP", "set_omp_package"]

FOMP: str = "__omp__"
_init_lock: Lock = Lock()
omp_packages: dict[str, Options] = {}


class Omp4pyFinder:
    """Meta path finder that enables OpenMP preprocessing for registered packages.

    This finder intercepts module imports and checks whether the top-level
    package name has been registered via `set_omp_package`. If so, it
    delegates module discovery to `PathFinder` and replaces the default
    source loader with `Omp4pyLoader`.

    The finder is intentionally selective and only affects packages that
    have been explicitly registered, leaving all other imports untouched.
    """

    @staticmethod
    def find_spec(
        name: str,
        import_path: Sequence[str] | None = None,
        target_module: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Locate and customize the module specification for registered packages.

        This method is invoked by Python's import system as part of the
        `sys.meta_path` protocol. If the module being imported belongs to
        a package registered in `omp_packages`, its loader is replaced
        with an `Omp4pyLoader` to enable import-time preprocessing.

        Args:
            name (str): Fully qualified name of the module being imported.
            import_path (Sequence[str] | None): Search path used to locate
                the module.
            target_module (types.ModuleType | None): Target module in case
                of a reload operation.

        Returns:
            importlib.machinery.ModuleSpec | None: A modified module
            specification using `Omp4pyLoader` for registered packages,
            or `None` if the module does not belong to a registered package.
        """
        if name.split(".", maxsplit=1)[0] not in omp_packages:
            return None
        spec: ModuleSpec | None = PathFinder.find_spec(name, import_path, target_module)
        if spec is not None and isinstance(spec.loader, SourceFileLoader) and spec.origin is not None:
            spec.loader = Omp4pyLoader(name, spec.origin)
        return spec


class Omp4pyLoader(SourceFileLoader):
    """Source loader that applies OpenMP preprocessing before compilation.

    This loader overrides the standard source-to-code compilation process to
    apply the `omp4py` preprocessor. Module source code is parsed into an AST,
    transformed using OpenMP directives, and then compiled into executable
    bytecode.

    A `__omp__` directory is used as the location for the transformed module
    cache. Its main purpose is to ensure that a separate `__pycache__`
    directory is created for the preprocessed code. This allows the compiled
    bytecode to be reused on subsequent imports, avoiding the need to
    preprocess the module again.

    Using a dedicated directory also prevents conflicts with the original
    module cache when `omp4py` is disabled, ensuring that normal imports
    continue to use the standard `__pycache__` without interference.
    """

    def source_to_code(self, data, path, *args, **kwargs) -> types.CodeType:  # type: ignore[override]  # noqa: ANN001, ANN002, ANN003
        """Transform module source code using the OpenMP preprocessor.

        The raw source code is parsed into an AST, passed through the
        `omp4py` preprocessor using the parameters associated with the
        registered package, and then compiled into a code object.

        Args:
            data: Raw source code of the module.
            path: Path to the module source file.
            *args: Additional positional arguments forwarded to the base
                implementation.
            **kwargs: Additional keyword arguments forwarded to the base
                implementation.

        Returns:
            types.CodeType: Compiled code object produced from the
            transformed AST.
        """
        from omp4py.core import preprocessor # Lazy import, only when needed (no cache available)
        opt: Options = omp_packages[self.name]
        module: ast.Module  = preprocessor.process_source(data, path, opt)
        return super().source_to_code(module, path, *args, **kwargs)

    def get_code(self, fullname: str) -> types.CodeType | None:
        """Load and compile a module using a virtual OpenMP-aware path.

        This method creates a virtual module path under the `__omp__`
        directory to allow transformed modules to coexist logically with
        their original source files. A temporary loader is used to
        redirect filesystem operations back to the original source file
        while preserving the transformed module identity.

        Args:
            fullname (str): Fully qualified name of the module.

        Returns:
            types.CodeType | None: Compiled code object for the transformed
            module, or `None` if the module cannot be loaded.
        """
        py_path: str = self.path
        omp_path: str = str(Path(py_path).parent / FOMP / Path(py_path).name)

        class DummyLoader(Omp4pyLoader):
            def get_data(self, path: str) -> bytes:
                return super().get_data(py_path if path == omp_path else path)

            def path_stats(self, path: str) -> Mapping[str, Any]:
                return super().path_stats(py_path if path == omp_path else path)

        opl: Omp4pyLoader = DummyLoader(fullname, omp_path)
        return SourceFileLoader.get_code(opl, fullname)


def set_omp_package(mod: ModuleType, opt: Options) -> ModuleType:
    """Register a package for import-time OpenMP preprocessing.

    This function marks a package as OpenMP-aware by associating it with
    a set of preprocessing parameters. Once registered, all modules
    imported from the package will be automatically transformed by the
    `omp4py` preprocessor during import.

    On the first registration, the global `Omp4pyFinder` is inserted at
    the beginning of `sys.meta_path` to activate the import hook.

    Args:
        mod (ModuleType): The package module to register. Typically, this
            is the result of importing the package's top-level module.
        opt (Options): Preprocessing options controlling how OpenMP
            directives are applied within the package.

    Returns:
        ModuleType: The same module that was passed in, allowing this
        function to be used inline during package initialization.
    """
    if len(omp_packages) == 0:
        with _init_lock:
            if len(omp_packages) == 0:
                sys.meta_path.insert(0, Omp4pyFinder())
    omp_packages[mod.__name__] = opt
    return mod
