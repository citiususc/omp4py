from omp4py.core import omp, omp4py_force_pure, __version__

if not omp4py_force_pure:
    import omp4py.cruntime as _runtime

    omp4py_compiled = _runtime.omp4py_compiled
else:
    omp4py_compiled = False

if omp4py_compiled:
    from omp4py.cruntime.api import *
else:
    import omp4py.runtime as _runtime
    from omp4py.runtime.api import *
