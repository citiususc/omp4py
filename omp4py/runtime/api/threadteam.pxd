from omp4py.runtime.lowlevel.numeric cimport pyint, pyint_array

cpdef void omp_set_num_threads(pyint num_threads)

cpdef pyint omp_get_num_threads()

cpdef pyint omp_get_max_threads()

cpdef pyint omp_get_thread_num()

cpdef bint omp_in_parallel()

cpdef void omp_set_dynamic(bint dynamic_threads)

cpdef bint omp_get_dynamic()

cpdef bint omp_get_cancellation()

cpdef void omp_set_nested(bint nested)

cpdef bint omp_get_nested()

ctypedef pyint omp_sched_t
cdef omp_sched_t omp_sched_static
cdef omp_sched_t omp_sched_dynamic
cdef omp_sched_t omp_sched_guided
cdef omp_sched_t omp_sched_auto
cdef omp_sched_t  omp_sched_monotonic

cdef pyint_array _omp_sched2run
cdef pyint _run2omp_sched_offset
cdef pyint_array _run2omp_sched

cpdef void omp_set_schedule(omp_sched_t kind, pyint chunk_size)

cpdef tuple omp_get_schedule()

cpdef pyint omp_get_thread_limit()

cpdef void omp_set_max_active_levels(pyint max_active_levels)

cpdef pyint omp_get_max_active_levels()

cpdef pyint omp_get_level()

cpdef pyint omp_get_ancestor_thread_num(pyint level)

cpdef pyint omp_get_team_size(pyint level)

cpdef pyint omp_get_active_level()
