from omp4py.core import omp as _omp, __version__
from omp4py.runtime.api import *


def omp(*args, **kwargs):
    kwargs['__omp4py_pure'] = True
    return _omp(*args, **kwargs)
