import sys
import os
from importlib.machinery import PathFinder, ExtensionFileLoader, SourceFileLoader
import omp4py.core.importer.cython

__all__ = []

class PureImporter:

    @staticmethod
    def find_spec(name, import_path, target_module):
        if "omp4py.runtime" in name:
            spec = PathFinder.find_spec(name, import_path, target_module)
            if isinstance(spec.loader, ExtensionFileLoader):
                pure_file: str = os.path.join(os.path.dirname(spec.origin), spec.name[len(spec.parent) + 1:] + '.py')
                spec.loader = SourceFileLoader(name, pure_file)
                spec.origin = pure_file

            return spec
        return None


sys.meta_path.insert(0, PureImporter())
if "cython" not in sys.modules:
    sys.modules["cython"] = omp4py.core.importer.cython
