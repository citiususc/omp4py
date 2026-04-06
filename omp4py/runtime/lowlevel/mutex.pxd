import cython
from omp4py.runtime.lowlevel.numeric cimport pyint


cdef extern from "Python.h":
    """
    #if PY_MINOR_VERSION > 13
        typedef struct { // Python private API
            uint8_t v;
        } _PyEvent;
        #define PyEvent _PyEvent
    #else
        #define Py_BUILD_CORE 1
        #include <internal/pycore_lock.h>
    #endif
    """
    ctypedef struct PyMutex:
        pass

    ctypedef struct PyEvent:
        pass


cdef class Mutex:
    cdef PyMutex _mutex

    @staticmethod
    cdef Mutex new()

    cdef void lock(self)

    cdef void unlock(self)

    cdef bint test(self)


cdef class RMutex:
    cdef PyMutex _mutex
    cdef cython.ulong _own
    cdef pyint _level

    @staticmethod
    cdef RMutex new()

    cdef void lock(self)

    cdef void unlock(self)

    cdef bint test(self)


cdef class Event:
    cdef PyEvent _event

    @staticmethod
    cdef Event new()

    cdef void wait(self)

    cdef void notify(self)
