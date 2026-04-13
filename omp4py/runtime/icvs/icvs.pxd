from omp4py.runtime.lowlevel.atomic cimport AtomicInt
from omp4py.runtime.lowlevel.numeric cimport pyint, pyint_array

cdef class Device:
    cdef pyint stacksize
    cdef bint wait_policy_active
    cdef pyint max_active_levels
    cdef AtomicInt threads_busy

    cdef Device copy(self)

cdef class Global:
    cdef bint cancel
    cdef pyint max_task_priority

    cdef Global copy(self)

cdef class PlacePartition:
    cdef pyint_array values
    cdef pyint_array partitions
    cdef object name

cdef class ImplicitTask:
    cdef PlacePartition place_partition

    cdef ImplicitTask copy(self)

cdef class RunSched:
    cdef bint monotonic
    cdef pyint type
    cdef pyint chunksize

    cdef RunSched copy(self)

cdef class Data:
    cdef bint dyn
    cdef bint nest
    cdef pyint_array nthreads
    cdef RunSched run_sched
    cdef pyint_array bind
    cdef bint bind_active
    cdef pyint thread_limit
    cdef pyint active_levels
    cdef pyint levels
    cdef pyint default_device
    ##
    cdef pyint team_size
    cdef pyint thread_num
    ##
    cdef Device device_vars
    cdef Global global_vars
    cdef ImplicitTask implicit_task_vars

    cdef Data copy(self)

cdef Data defaults