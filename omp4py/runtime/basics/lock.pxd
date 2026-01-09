from omp4py.runtime.basics.types cimport *


cdef extern from "Python.h":
    """
    typedef struct { // Python private API
        uint8_t v;
    } _PyEvent;
    #define PyEvent _PyEvent
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
    cdef pyint _own
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
