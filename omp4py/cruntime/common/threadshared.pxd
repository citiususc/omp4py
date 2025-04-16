from omp4py.cruntime.basics cimport atomic, lock
from omp4py.cruntime.basics.types cimport *

cdef class _SharedEntry:
    cdef object obj
    cdef pyint tp
    cdef atomic.AtomicObject next

cdef class SharedContext:
    cdef  _SharedEntry _head
    cdef _SharedEntry _tail

    @staticmethod
    cdef SharedContext new()

    cdef object push(self, pyint tp, object value)

    cdef object get(self, pyint tp)

    cdef void move_last(self)

    cdef SharedContext __copy__(self)

cdef class _QueueEntry:
    cdef object obj
    cdef atomic.AtomicFlag free
    cdef lock.Event wait_event
    cdef atomic.AtomicObject next

cdef class TaskQueue:
    cdef _QueueEntry _head
    cdef _QueueEntry _tail
    cdef _QueueEntry _history

    @staticmethod
    cdef TaskQueue new()

    cdef void add(self, object value)

    cdef object take(self)

    cdef void set_history(self)

    cdef object history_take(self)

    cdef TaskQueue __copy__(self)

cdef class SharedFactory:
    cdef SharedContext _context
    cdef TaskQueue _tasks
    cdef lock.Mutex lock_mutex

    cdef SharedContext context(self)

    cdef TaskQueue task_queue(self)
