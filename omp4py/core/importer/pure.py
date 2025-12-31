import sys
import os
import importlib.machinery


class PureImporter:

    @staticmethod
    def find_spec(name, import_path, target_module):
        if "omp4py.runtime" in name:
            from importlib.machinery import PathFinder
            spec = PathFinder.find_spec(name, import_path, target_module)
            if isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                pure_file: str = os.path.join(os.path.dirname(spec.origin), spec.name[len(spec.parent) + 1:] + '.py')
                spec.loader = importlib.machinery.SourceFileLoader(name, pure_file)
                spec.origin = pure_file

            return spec
        return None


sys.meta_path.insert(0, PureImporter())
