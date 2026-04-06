from omp4py.runtime.icvs cimport Data
from omp4py.runtime.lowlevel cimport threadlocal # omp4py_threadlocal_get

cdef class TaskContext:
    cdef Data icvs

cdef void context_init()

cdef extern from *:
    # Redefine omp_ctx as a direct call to C function.
    cdef TaskContext omp_ctx "omp4py_threadlocal_get"()
