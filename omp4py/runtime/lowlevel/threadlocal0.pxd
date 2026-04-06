"""Definition of the `omp4py` thread-local runtime variable.

This module declares and defines the C thread-local variable
`omp4py_local0`, which is used by the compiled `omp4py` runtime to store
the current thread-specific state.

Each module defines its own static thread-local variable, `omp4py_local`,
which stores a pointer to `omp4py_local0`. This pointer is initialized
via `omp4py_local0_ptr`, allowing each module to maintain a local copy
of the reference.

This design provides direct access to the thread-local variable without
incurring function call overhead, while also avoiding cross-module
symbol library linking.
"""
from cpython.object cimport PyObject

cdef extern from *:
    """
    thread_local PyObject* omp4py_local0 = NULL;

    PyObject** omp4py_local0_ptr(){
        return &omp4py_local0;
    }

    """
    cdef PyObject** omp4py_local0_ptr()
