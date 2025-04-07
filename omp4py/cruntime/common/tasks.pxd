from omp4py.cruntime.basics cimport array, lock, atomic
from omp4py.cruntime.common cimport controlvars, threadshared
from omp4py.cruntime.basics.types cimport *

cdef pyint ParallelTaskID
cdef pyint ForTaskID
cdef pyint TeamsTaskID
cdef pyint SectionsTaskID

cdef class Task:
    cdef Task parent
    cdef ParallelTask parallel
    cdef TeamsTask teams
    cdef controlvars.ControlVars cvars

cdef class ParallelTask(Task):
    cdef threadshared.SharedContext context
    cdef threadshared.TaskQueue queue
    cdef lock.Mutex lock_mutex
    cdef lock.Barrier lock_barrier

    @staticmethod
    cdef ParallelTask new(controlvars.ControlVars cvars, threadshared.SharedFactory shared)

ctypedef void(*kind_f)(ForTask, array.iview)

cdef class ForTask(Task):
    cdef kind_f kind
    cdef pyint collapse
    cdef bint monotonic
    cdef bint first_chunk
    cdef pyint iters
    cdef pyint chunk
    cdef pyint count
    cdef pyint step
    cdef pyint current_chunk
    cdef atomic.AtomicInt shared_count

    @staticmethod
    cdef ForTask new(controlvars.ControlVars cvars)

cdef class TeamsTask:
    pass

cdef class CustomTask:
    cdef object f