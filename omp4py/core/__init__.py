
from omp4py.core.options import Options

if Options._new_core:
    from omp4py.core.api import omp
else:
    import omp4py.core.processor as _
    from omp4py.core.parser_old import omp

if Options.pure:
    import omp4py.core.imports.pure as _

__all__ = ["omp"]
