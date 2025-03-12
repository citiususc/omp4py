from omp4py.cruntime.basics.types cimport *

cdef class BaseEnum:
    cdef pyint _value

cdef class omp_sched_t(BaseEnum):
    pass

cdef omp_sched_t omp_sched_static
cdef omp_sched_t omp_sched_dynamic
cdef omp_sched_t omp_sched_guided
cdef omp_sched_t omp_sched_auto
cdef omp_sched_t omp_sched_monotonic
cdef dict[str, str] omp_sched_names
