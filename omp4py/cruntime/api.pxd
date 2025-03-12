from omp4py.cruntime.common.enums cimport omp_sched_t
from omp4py.cruntime.basics.types cimport pyint

#######################################################################################################################
########################################## Parallel Region Support Routines ###########################################
#######################################################################################################################

cpdef void omp_set_num_threads(pyint num_threads)

cpdef pyint omp_get_num_threads()

cpdef pyint omp_get_thread_num()

cpdef pyint omp_get_max_threads()

cpdef pyint omp_get_thread_limit()

cpdef bint omp_in_parallel()

cpdef void omp_set_dynamic(bint dynamic_threads)

cpdef bint omp_get_dynamic()

cpdef void omp_set_schedule(omp_sched_t kind, pyint chunk_size= *)

cdef class omp_get_schedule_t:
    cdef omp_sched_t kind
    cdef pyint chunk_size

    @staticmethod
    cdef _new(omp_sched_t kind, pyint chunk_size)

cpdef omp_get_schedule_t omp_get_schedule()

cpdef pyint omp_get_supported_active_levels()

cpdef void omp_set_max_active_levels(pyint max_levels)

cpdef pyint omp_get_max_active_levels()

cpdef pyint omp_get_level()

cpdef pyint omp_get_ancestor_thread_num(pyint level)

cpdef pyint omp_get_team_size(pyint level)

cpdef pyint omp_get_active_level()

#######################################################################################################################
################################################ Teams Region Routines ################################################
#######################################################################################################################
cpdef pyint omp_get_num_teams()

cpdef void omp_set_num_teams(pyint num_teams)

cpdef pyint omp_get_team_num()

cpdef pyint omp_get_max_teams()

cpdef pyint omp_get_teams_thread_limit()

cpdef void omp_set_teams_thread_limit(pyint thread_limit)
