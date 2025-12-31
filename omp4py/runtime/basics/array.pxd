from omp4py.runtime.basics.types cimport pyint, pyfloat
from cpython.array cimport array


ctypedef pyint[:] iview
ctypedef pyfloat[:] fview

cdef array[pyint] new_int(pyint n)

cdef array[pyfloat] new_float(pyint n)

cdef array[pyint] int_from(list[pyint] elems)

cdef array[pyfloat] float_from(list[pyfloat] elems)

cdef iview as_iview(elems: array[pyint])

cdef fview as_fview(elems: array[pyfloat])

cdef array copy(array elems)