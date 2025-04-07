from omp4py.cruntime.basics cimport atomic, lock
from omp4py.cruntime.basics.types cimport *

cdef class _SharedEntry:
    cdef object obj
    cdef pyint tp
    cdef atomic.AtomicObject next

cdef class SharedContext:
    cdef  _SharedEntry _head
    cdef _SharedEntry _tail

    cdef object push(self, pyint tp, object value)

    cdef void pop(self)

    cdef bint has(self)

cdef class _QueueEntry:
    cdef object obj
    cdef atomic.AtomicFlag free
    cdef atomic.AtomicObject next

cdef class TaskQueue:
    cdef _QueueEntry _head
    cdef _QueueEntry _tail

    cdef void add(self, object value)

    cdef object take(self)

cdef class SharedFactory:
    cdef _SharedEntry _context
    cdef _QueueEntry _tasks
    cdef lock.Mutex lock_mutex
    cdef lock.Barrier lock_barrier

    cdef SharedContext context(self)

    cdef TaskQueue task_queue(self)
