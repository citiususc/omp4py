"""Compiled thread-local storage used by the `omp4py` runtime.

This module provides the compiled implementation of the thread-local
storage utilities used by the `omp4py` runtime. Instead of relying on
Python's `threading.local`, it uses a native C `thread_local` variable
to store a pointer to a Python object associated with the current thread.

The `ThreadLocal` class exposes the same interface as the pure Python
implementation, but internally reads and writes the thread-local value
directly through the C variable `omp4py_local`. This approach avoids the
overhead of `threading.local` and provides faster access when the runtime
is compiled with Cython.
"""

from cpython.object cimport PyObject

cdef extern from *: # https://stackoverflow.com/questions/18298280/how-to-declare-a-variable-as-thread-local-portably
    """
    #ifndef thread_local
    # if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
    #  define thread_local _Thread_local
    # elif defined _WIN32 && ( defined _MSC_VER || defined __ICL || defined __DMC__ || defined __BORLANDC__ )
    #  define thread_local __declspec(thread)
    /* note that ICC (linux) and Clang are covered by __GNUC__ */
    # elif defined __GNUC__ || defined __SUNPRO_C || defined __hpux || defined __xlC__
    #  define thread_local __thread
    # else
    #  error "Cannot define thread_local"
    # endif
    #endif

    extern thread_local PyObject* omp4py_local;
    """
    PyObject *omp4py_local

cdef class ThreadLocal:

    @staticmethod
    cdef inline void set(value: object):
        global omp4py_local
        omp4py_local = <PyObject *> value
        ThreadLocal.default_set(value)

    @staticmethod
    cdef inline object get():
        return <object>omp4py_local