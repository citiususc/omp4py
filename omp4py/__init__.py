from omp4py.core import omp
import omp4py.cruntime as _runtime

if _runtime.omp4py_compiled:
    from omp4py.cruntime.api import *
else:
    import omp4py.runtime as _runtime
    from omp4py.runtime.api import *
