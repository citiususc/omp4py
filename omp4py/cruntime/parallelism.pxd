from omp4py.cruntime.basics.types cimport *

#TODO object can be a fused type, importante for explicit tasks
cpdef void parallel_run(object f, bint c_if, str c_message, tuple[pyint, ...] c_nthreads, pyint c_safesync, str c_severity)
