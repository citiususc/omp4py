"""Compiled numeric type aliases for the `omp4py` runtime.

This module defines numeric type aliases that are used when the `omp4py`
runtime is compiled with Cython and the original Python-level aliases
do not exist. These aliases ensure consistent type annotations and
numeric operations across both the pure Python and compiled versions
of the runtime.

This approach prevents runtime errors when code written for the
Python runtime is executed against the compiled runtime.
"""

_pyint_array_template= array.array("q", [])
_pyfloat_array_template = array.array("d", [])

cdef long long _pyint_template = 0
cdef long long[:] _pyint_mv_template = _pyint_array_template[:]
cdef double _pyfloat_template = 0
cdef double[:] _pyfloat_mv_template = _pyfloat_array_template[:]

globals()["pyint"] = type(_pyint_template)
globals()["pyint_array"] = type(_pyint_mv_template)
globals()["pyfloat"] = type(_pyfloat_template)
globals()["pyfloat_array"] = type(_pyfloat_mv_template)
