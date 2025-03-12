from omp4py.cruntime.basics.types cimport *
cimport omp4py.cruntime.basics.array as array

cdef class ScheduleVar:
    cdef bint monotonic
    cdef pyint kind
    cdef pyint chunk

    cdef ScheduleVar set(self, bint monotonic, pyint kind, pyint chunk)

    @staticmethod
    cdef pyint kind_int(kind: str)

    cdef __copy__(self)

cdef class GlobalVars:
    cdef str available_devices
    cdef bint cancel
    cdef bint debug
    cdef bint display_affinity
    cdef pyint max_task_priority
    cdef pyint num_devices
    cdef str target_offload

    cdef default(self)

    cdef __copy__(self)


cdef class DataEnvVars:
    cdef pyint active_levels
    cdef str bind
    cdef str default_device
    cdef bint dyn
    cdef bint explicit_task
    cdef bint final_task
    cdef pyint free_agent_thread_limit
    cdef bint free_agent
    cdef pyint league_size
    cdef pyint levels
    cdef pyint max_active_levels
    cdef array.iview nthreads
    cdef ScheduleVar run_sched
    cdef pyint structured_thread_limit
    cdef pyint team_generator
    cdef pyint team_num
    cdef pyint team_size
    cdef pyint thread_limit
    cdef pyint thread_num

    cdef default(self)

    cdef __copy__(self)


cdef class DeviceVars:
    cdef str affinity_format
    cdef pyint device_num
    cdef pyint nteams
    cdef pyint num_procs
    cdef pyint stacksize
    cdef pyint teams_thread_limit
    cdef str wait_policy

    cdef default(self)

    cdef __copy__(self)


cdef class ITaskVars:
    cdef str def_allocator
    cdef str place_assignment

    cdef default(self)

    cdef __copy__(self)

cdef class ControlVars:
    cdef GlobalVars global_
    cdef DataEnvVars dataenv
    cdef DeviceVars device
    cdef ITaskVars itask

    cdef default(self)

    cdef __copy__(self)
