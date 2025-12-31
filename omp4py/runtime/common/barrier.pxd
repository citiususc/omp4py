from omp4py.runtime.basics cimport lock, atomic
from omp4py.runtime.common cimport tasks, threadshared

cdef class BarrierContext:
    cdef threadshared.SharedContext ctx
    cdef atomic.AtomicInt count

    @staticmethod
    cdef BarrierContext new()

cdef class BarrierShared:
    cdef bint notify_event
    cdef lock.Event event

    @staticmethod
    cdef BarrierShared new()

cdef void task_barrier(tasks.Task task)

cdef void task_notify(tasks.Task task)
