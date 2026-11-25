cimport cython
from cpython.list cimport PyList_GET_ITEM

from omp4py.runtime.lowlevel.atomic cimport AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.mutex cimport Event
from omp4py.runtime.lowlevel.numeric cimport pyint


@cython.final
cdef class Sleepers:
    cdef list[AtomicObject] _slots
    cdef AtomicInt _sleeping
    cdef pyint _n

    @staticmethod
    cdef Sleepers new(pyint n)

    cdef Event prepare(self, pyint tid)

    cdef bint cancel(self, pyint tid)

    cdef void wait(self, Event event)

    cdef bint wake_one(self, pyint tid)

    cdef pyint wake_all(self)

    cdef bint any_sleeping(self) noexcept


cdef inline void _slot_set(Sleepers sleepers, pyint i, Event event) noexcept:
    (<AtomicObject> PyList_GET_ITEM(sleepers._slots, i)).set(event)  # Fast version of sleepers._slots[i].set(event)


cdef inline Event _slot_take(Sleepers sleepers, pyint i):
    return <Event> (<AtomicObject> PyList_GET_ITEM(sleepers._slots, i)).exchange(None)  # Fast version of ...[i].exchange(None)
