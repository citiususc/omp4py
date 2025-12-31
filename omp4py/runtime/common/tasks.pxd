from omp4py.runtime.basics cimport array, lock, atomic
from omp4py.runtime.common cimport controlvars, threadshared
from omp4py.runtime.basics.types cimport *

cdef pyint ParallelTaskID
cdef pyint ForTaskID
cdef pyint TeamsTaskID
cdef pyint SectionsTaskID
cdef pyint SingleTaskID
cdef pyint BarrierTaskID
cdef pyint CustomTaskID

cdef class Task:
    cdef Task parent
    cdef ParallelTask parallel
    cdef TeamsTask teams
    cdef controlvars.ControlVars cvars

cdef class ParallelTask(Task):
    cdef threadshared.SharedContext context
    cdef threadshared.TaskQueue queue
    cdef lock.Mutex lock_mutex

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

cdef class TeamsTask(Task):
    pass

cdef class SingleTask(Task):
    cdef atomic.AtomicFlag executed

    @staticmethod
    cdef SingleTask new(controlvars.ControlVars cvars, atomic.AtomicFlag executed)


cdef class BarrierTask(Task):
    cdef pyint parties
    cdef atomic.AtomicInt count
    cdef threadshared.SharedContext context

    @staticmethod
    cdef BarrierTask new(controlvars.ControlVars cvars, pyint parties, threadshared.SharedContext context,
                         atomic.AtomicInt count)

cdef class CustomTask(Task):
    cdef object f
    cdef lock.Event wait_event

    @staticmethod
    cdef CustomTask new(controlvars.ControlVars cvars, object f)