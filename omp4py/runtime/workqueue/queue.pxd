cimport cython
from cpython.list cimport PyList_GET_ITEM, PyList_SET_ITEM
from cpython.object cimport PyObject
from cpython.ref cimport Py_DECREF, Py_INCREF, Py_XDECREF

from omp4py.runtime.lowlevel.atomic cimport AtomicInt
from omp4py.runtime.lowlevel.mutex cimport Mutex
from omp4py.runtime.lowlevel.numeric cimport CACHE_LINE, pyint


@cython.final
cdef class TaskQueue:
    # Read by the thieves, written rarely
    cdef list _items
    cdef pyint _mask
    cdef pyint _max_size
    cdef pyint _split
    cdef AtomicInt _head
    cdef AtomicInt _shared
    cdef Mutex _lock
    # The owner writes the next ones at every push: a cache line of their own
    cdef char _padding[CACHE_LINE]
    cdef pyint _tail
    cdef pyint _head_cache
    # and the objects that are allocated after this one do not share it either
    cdef char _end_padding[CACHE_LINE]

    @staticmethod
    cdef TaskQueue new(pyint initial, pyint maximum)

    cdef bint push(self, object task) except -1

    cdef object pop(self)

    cdef object steal(self)

    cdef pyint size(self) noexcept

    cdef pyint stealable(self) noexcept

    cdef void _release(self) noexcept

    cdef bint _make_room(self, pyint tail) except -1

    cdef void _grow(self) except *


cdef inline void _slot_put(TaskQueue queue, pyint i, object task) noexcept:
    # Fast version of queue._items[i & queue._mask] = task. The slot is always empty (None).
    cdef pyint slot = i & queue._mask
    cdef PyObject* old = PyList_GET_ITEM(queue._items, slot)
    Py_INCREF(task)
    PyList_SET_ITEM(queue._items, slot, task)
    Py_XDECREF(old)


cdef inline object _slot_take(TaskQueue queue, pyint i):
    # Fast version of: task = queue._items[i & queue._mask]; queue._items[i & queue._mask] = None
    cdef pyint slot = i & queue._mask
    cdef PyObject* old = PyList_GET_ITEM(queue._items, slot)
    cdef object task = <object> old
    Py_INCREF(None)
    PyList_SET_ITEM(queue._items, slot, None)
    Py_DECREF(task)  # the reference that the ring had
    return task
