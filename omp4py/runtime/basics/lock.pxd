from omp4py.runtime.basics.types cimport *


cdef extern from *:
    """
    #define Py_BUILD_CORE 1
    #include <internal/pycore_lock.h>

    int PyMutex_LockFast_(PyMutex *m){
        #if PY_MINOR_VERSION < 14
            return PyMutex_LockFast((uint8_t*)m);
        #else
            return PyMutex_LockFast(m);
        #endif
    }
    """
    ctypedef struct PyMutex:
        pass

    void PyMutex_Lock(PyMutex *m)

    int PyMutex_LockFast_(PyMutex *m)

    void PyMutex_Unlock(PyMutex *m)

    ctypedef struct PyEvent:
        pass

    void _PyEvent_Notify(PyEvent *evt)

    void PyEvent_Wait(PyEvent *evt)



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
