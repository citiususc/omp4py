from omp4py.runtime.basics.types cimport *

cdef bint has_storage()
cdef object get_storage()
cdef void set_storage(object value)
cdef pyint thread_id()