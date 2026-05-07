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

import sys
import types
from collections.abc import Sequence
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from pathlib import Path
from threading import Lock

from omp4py.core.options import Options

__all__ = ["OMP_FOLDER", "set_omp_package"]

OMP_FOLDER: str = "__omp__"
_init_lock: Lock = Lock()
omp_packages: dict[str, Options] = {}


class Omp4pyFinder:
    """Meta pathfinder that enables OpenMP preprocessing for registered packages.

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
        if name.rsplit(".", maxsplit=1)[0] not in omp_packages:
            return None
        spec: ModuleSpec | None = PathFinder.find_spec(name, import_path, target_module)
        if spec is not None and isinstance(spec.loader, SourceFileLoader) and spec.origin is not None:
            spec.loader = Omp4pyLoader(name, spec.origin)
            spec.origin = str(Path(spec.origin).parent / OMP_FOLDER / Path(spec.origin).name)
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
        if isinstance(path, bytes):
            path = path.decode()
        if isinstance(data, bytes):
            data = data.decode()

        omp_path: str = str(Path(path).parent / OMP_FOLDER / Path(path).name)
        if isinstance(data, str):
            from omp4py.core import preprocessor  # noqa: PLC0415  Lazy import, only when needed (no cache available)

            opt: Options = omp_packages[self.name.rsplit(".", maxsplit=1)[0]]
            data = preprocessor.process_source(data, omp_path, opt)
            path = omp_path
        return super().source_to_code(data, path, *args, **kwargs)

    def get_data(self, path: str) -> bytes:
        """Retrieve module data, redirecting cache accesses to the OpenMP cache.

        This method overrides the default data loading mechanism to intercept
        accesses to files inside a `__pycache__` directory. When such a path is
        detected, it is transparently redirected to the equivalent location
        within the `__omp__` cache directory.

        The original source files remain untouched and are loaded normally.
        Only cached bytecode and related files are affected, ensuring that
        Python reuses the OpenMP-preprocessed cache instead of the standard
        `__pycache__`.

        Args:
            path (str): Filesystem path requested by the import system.

        Returns:
            bytes: Raw file contents, loaded from the OpenMP cache when the
            request targets a cache file, or from the original location
            otherwise.
        """
        target = Path(path)
        if target.parent.name == "__pycache__":
            if omp_packages[self.name.rsplit(".", maxsplit=1)[0]].ignore_cache:
                msg = "Cache directory is detected and ignored"
                raise OSError(msg)
            target = Path(*target.parts[:-2], OMP_FOLDER, *target.parts[-2:])

        return super().get_data(str(target))


def set_omp_package(pkg: str, opt: Options) -> None:
    """Register a package for import-time OpenMP preprocessing.

    This function marks a package as OpenMP-aware by associating it with
    a set of preprocessing parameters. Once registered, all modules
    imported from the package will be automatically transformed by the
    `omp4py` preprocessor during import.

    On the first registration, the global `Omp4pyFinder` is inserted at
    the beginning of `sys.meta_path` to activate the import hook.

    Args:
        pkg (str): The `__package__` value of `__init__.py` module
            where the preprocessor will be applied.
        opt (Options): Preprocessing options controlling how OpenMP
            directives are applied within the package.
    """
    if len(omp_packages) == 0:
        with _init_lock:
            if len(omp_packages) == 0:
                sys.meta_path.insert(0, Omp4pyFinder())
    omp_packages[pkg] = opt
