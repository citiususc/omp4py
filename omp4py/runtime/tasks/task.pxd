from cpython.object cimport Py_TYPE

from omp4py.runtime.icvs cimport Data
from omp4py.runtime.lowlevel.atomic cimport AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.numeric cimport pyint


cdef inline bint same_class(object obj1, object obj2):
    return Py_TYPE(obj1) == Py_TYPE(obj2)

cdef class Task:
    cdef SharedContext shared
    cdef Data icvs
    cdef Task _return_to
    cdef Barrier barrier

    cdef Task _newTask(self, SharedContext shared, Data icvs)

cdef class SharedItem:
    cdef object value
    cdef AtomicObject next

    @staticmethod
    cdef SharedItem new(object value)


cdef class SharedContext:
    cdef AtomicObject _current

    @staticmethod
    cdef SharedContext new()

    cdef SharedContext mirror(self)

    cdef object sync(self, object obj)

cdef class Barrier:
    cdef pyint _parties
    cdef AtomicInt _waiting
    cdef AtomicObject _event
    cdef pyint _gen

    @staticmethod
    cdef Barrier new(pyint parties)

    cdef bint wait(self)

    cdef void interrupt(self)