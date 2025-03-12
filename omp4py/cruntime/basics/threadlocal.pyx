from threading import local
from cpython.object cimport PyObject

_storage: local = local()

cdef extern from '<threads.h>':
    """
    thread_local PyObject* omp_thread = NULL;
    """
    PyObject *omp_thread


cdef bint has_storage():
    return omp_thread != NULL

cdef object get_storage():
    return <object> omp_thread

cdef void set_storage(object value):
    global omp_thread
    omp_thread = <PyObject *> value
    setattr(_storage, 'omp', value)
