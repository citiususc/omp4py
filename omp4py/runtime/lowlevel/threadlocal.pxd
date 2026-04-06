"""Compiled thread-local storage used by the `omp4py` runtime.

This module provides the compiled implementation of the thread-local
storage utilities used by the `omp4py` runtime. Instead of relying on
Python's `threading.local`, it uses a native C `thread_local` variable
to store a pointer to a Python object associated with the current thread.

To avoid cross-module symbol linking, the thread-local variable is
declared as `static` in each module. Each module maintains a thread-local
pointer to the shared `thread_local` variable. On first access, this
pointer is initialized via `threadlocal_link`; if the shared variable has
not yet been initialized, `threadlocal_init` is called.

This design avoids the overhead of `threading.local` and provides faster
access when the runtime is compiled with Cython.
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

    static thread_local PyObject** omp4py_local = NULL;

    static CYTHON_INLINE PyObject **omp4py_threadlocal_cinit(void);
    // Efficient context access: C function using Cython only on first access
    inline PyObject * omp4py_threadlocal_get() {
        if (omp4py_local == NULL){
            omp4py_local = omp4py_threadlocal_cinit();
        }
        #ifndef Py_GIL_DISABLED // Acquire GIL on non-free-threaded builds
            PyGILState_STATE s = PyGILState_Ensure(); Py_XINCREF(*omp4py_local); PyGILState_Release(s);
        #else
            Py_XINCREF(*omp4py_local);
        #endif
        return *omp4py_local;
    }
    """
    cdef object omp4py_threadlocal_get()


cdef void threadlocal_default_set(object value)

cdef inline void threadlocal_set(object value):
    threadlocal_default_set(value)
    threadlocal_link()[0] = <PyObject *> value

cdef inline object threadlocal_get():
    return omp4py_threadlocal_get()

cdef void threadlocal_init_none()


cdef void(*threadlocal_init)() except *

cdef PyObject**(*threadlocal_link)() except *

# Reduce C function complexity by implementing the slow path of omp4py_threadlocal_get in Cython
cdef inline PyObject** threadlocal_cinit "omp4py_threadlocal_cinit"() nogil:
    cdef PyObject** omp4py_local
    with gil:
        omp4py_local = threadlocal_link()
        if  omp4py_local[0] == NULL:
            threadlocal_init()
    return omp4py_local
