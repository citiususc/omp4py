"""Definition of the `omp4py` thread-local runtime variable.

This module declares and defines the C thread-local variable
`omp4py_local` used by the compiled `omp4py` runtime to store the
current thread-local value.

The variable is initialized to `PyNone` so that it always contains a
valid Python object when accessed. This definition is included when
compiling Python modules with Cython, ensuring that the thread-local
storage used by `ThreadLocal` has a proper initial value in each thread.

This small module exists solely to provide the definition of the
thread-local symbol so it can be safely referenced from other compiled
runtime modules.
"""
cdef extern from *:
    """
    thread_local PyObject* omp4py_local = PyNone;
    """