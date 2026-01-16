"""Pure-Python import hook for forcing source-based omp4py runtime loading.

This module defines a custom importer that intercepts Python's import
machinery to prevent loading compiled native extensions for the
`omp4py.runtime` package. Instead, it forces the import system to load
the corresponding pure-Python source files (`.py`), even when compiled
extensions (e.g., `.so`, `.pyd`) are available.

The importer works by registering itself at the beginning of
`sys.meta_path` and selectively overriding the loader for modules under
`omp4py.runtime` when a compiled extension loader is detected.
"""

import sys
import types
from collections.abc import Sequence
from importlib.machinery import ExtensionFileLoader, ModuleSpec, PathFinder, SourceFileLoader
from pathlib import Path

__all__ = []


class PureImport:
    """Meta path import that forces pure-Python imports for `omp4py` runtime modules.

    This class implements a custom import hook that intercepts module resolution
    for `omp4py.runtime` modules. When a compiled extension module is detected,
    it replaces the default extension loader with a source file loader pointing
    to the corresponding `.py` file.
    """

    @staticmethod
    def find_spec(name: str, import_path: Sequence[str] | None = None, target_module: types.ModuleType | None = None) \
            -> ModuleSpec | None:
        """Find and customize the module specification for `omp4py` runtime modules.

        This method is invoked by Python's import machinery as part of the
        `sys.meta_path` protocol. If the requested module belongs to
        `omp4py.runtime` and its resolved loader corresponds to a compiled
        extension, the loader is replaced with a `SourceFileLoader` pointing
        to the pure-Python source file.

        Args:
            name (str): Fully qualified name of the module being imported.
            import_path (list[str] | None): Search path for module resolution.
            target_module (module | None): Target module, if this is a reload
                operation.

        Returns:
            importlib.machinery.ModuleSpec | None: A modified module specification
            that forces source-based loading for `omp4py` runtime modules, or
            `None` if the importer does not apply.
        """
        if "omp4py.runtime" in name:
            spec: ModuleSpec | None = PathFinder.find_spec(name, import_path, target_module)
            if spec is not None and isinstance(spec.loader, ExtensionFileLoader) and spec.origin is not None:
                pure_file: str = str(Path(spec.origin).parent / (name.split(".")[-1] + ".py"))
                spec.loader = SourceFileLoader(name, pure_file)
                spec.origin = pure_file

            return spec
        return None


sys.meta_path.insert(0, PureImport())
if "cython" not in sys.modules:
    from omp4py.core.imports import cython
    sys.modules["cython"] = cython
