"""Compiled synchronization primitives used by the `omp4py` runtime.

This module provides the compiled implementation of the basic
synchronization primitives used by the `omp4py` runtime. It relies on
low-level synchronization objects from CPython's internal runtime,
including `PyMutex` and `PyEvent`, to implement efficient thread
synchronization.

The classes defined in this module (`Mutex`, `RMutex`, and `Event`)
expose the same interface as the pure Python runtime implementation,
but internally use CPython primitives instead of the `threading`
module. This allows synchronization operations to execute with lower
overhead in the compiled runtime.

These implementations are used when `omp4py` is compiled with Cython,
providing faster synchronization while keeping the same public API as
the pure Python runtime.
"""

cimport omp4py.runtime.lowlevel.threadlocalh #thread_local keyword

cdef extern from *:
    """
    #undef PyEvent
    #define Py_BUILD_CORE 1
    #include <internal/pycore_lock.h>

    int PyMutex_LockFast_(PyMutex *m){
        #if PY_MINOR_VERSION < 14
            return PyMutex_LockFast((uint8_t*)m);
        #else
            return PyMutex_LockFast(m);
        #endif
    }

    PyMutex PyMutex_Init_(){
        return (PyMutex){0};
    }

    PyEvent PyEvent_Init_(){
        return (PyEvent){0};
    }

    thread_local unsigned long omp4py_thread_native_cache = -1;

    unsigned long omp4py_thread_native_cache_get(){
        if (omp4py_thread_native_cache != -1){ return omp4py_thread_native_cache;}
        omp4py_thread_native_cache = PyThread_get_thread_native_id();
        return omp4py_thread_native_cache;
    }

    """

    PyMutex PyMutex_Init_()

    void PyMutex_Lock(PyMutex *m)

    int PyMutex_LockFast_(PyMutex *m)

    void PyMutex_Unlock(PyMutex *m)

    int PyMutex_IsLocked(PyMutex *m)

    PyEvent PyEvent_Init_()

    void _PyEvent_Notify(PyEvent *evt)

    void PyEvent_Wait(PyEvent *evt)

    unsigned long thread_id "omp4py_thread_native_cache_get"()


cdef class Mutex:
    @staticmethod
    cdef Mutex new():
        m: Mutex = Mutex.__new__(Mutex)
        m._mutex = PyMutex_Init_()
        return m

    def __init__(self):
        raise ValueError("use new() class method")

    cdef void lock(self):
        PyMutex_Lock(&self._mutex)

    cdef void unlock(self):
        PyMutex_Unlock(&self._mutex)

    cdef bint test(self):
        return PyMutex_LockFast_(&self._mutex)

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False


cdef class RMutex:
    @staticmethod
    cdef RMutex new():
        m: RMutex = RMutex.__new__(RMutex)
        m._mutex = PyMutex_Init_()
        m._own = -1
        m._level = 0
        return m

    def __init__(self):
        raise ValueError("use new() class method")

    cdef void lock(self):
        if self._own == -1:
            self._own = thread_id()
            self._level += 1
            PyMutex_Lock(&self._mutex)
        elif self._own != thread_id():
            self._level += 1

    cdef void unlock(self):
        if self._own == thread_id():
            self._level -= 1
            if self._level == 0:
                PyMutex_Unlock(&self._mutex)

    cdef bint test(self):
        if self._own == -1:
            result: int = PyMutex_LockFast_(&self._mutex)
            if result:
                self._level += 1
            return result
        elif self._own != thread_id():
            self._level += 1
            return True

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False


cdef class Event:
    @staticmethod
    cdef Event new():
        e: Event = Event.__new__(Event)
        e._event = PyEvent_Init_()
        return e

    def __init__(self):
        raise ValueError("use new() class method")

    cdef void wait(self):
        PyEvent_Wait(&self._event)

    cdef void notify(self):
        _PyEvent_Notify(&self._event)
