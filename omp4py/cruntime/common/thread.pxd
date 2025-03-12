cimport omp4py.cruntime.common.tasks as tasks
cimport omp4py.cruntime.common.controlvars as controlvars

cdef class OmpThread:
    cdef tasks.Task task
    cdef tasks.ParallelTask parallel
    cdef tasks.TeamsTask teams

    cdef OmpThread set_task(self, tasks.Task task)

    cdef void pop_task(self)

    cdef void set_parallel(self, tasks.ParallelTask task)

    cdef void set_teams(self, tasks.TeamsTask task)

cdef OmpThread init(tasks.Task task)

cdef OmpThread current()

cdef controlvars.ControlVars cvars()