import os

if os.environ.get("_OMP4PY_NEW_CORE"):
    from omp4py.core.api import omp
else:
    import omp4py.core.processor as _
    from omp4py.core.parser_old import omp

__all__ = ["omp"]
