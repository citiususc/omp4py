cimport cython
from cpython.object cimport Py_TYPE, PyTypeObject

from omp4py.runtime.icvs cimport Data
from omp4py.runtime.lowlevel.atomic cimport AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.numeric cimport pyint


cdef inline bint same_class(object obj1, object obj2):
    return Py_TYPE(obj1) == Py_TYPE(obj2)

cdef inline bint instanceof(object obj, type cls):
    return Py_TYPE(obj) == <PyTypeObject*>cls

cdef class Task:
    cdef SharedContext shared
    cdef Data icvs
    cdef Task _return_to
    cdef Barrier barrier

    cdef Task _new_task(self, SharedContext shared, Data icvs, Barrier barrier)

@cython.final
cdef class SharedItem:
    cdef object value
    cdef AtomicObject next

    @staticmethod
    cdef SharedItem new(object value)


@cython.final
cdef class SharedContext:
    cdef AtomicObject _current

    @staticmethod
    cdef SharedContext new()

    cdef SharedContext mirror(self)

    cdef object get(self, type cls)

    cdef object sync(self, object obj)


@cython.final
cdef class Barrier:
    cdef pyint _parties
    cdef AtomicInt _waiting
    cdef AtomicObject _event
    cdef pyint _gen

    @staticmethod
    cdef Barrier new(pyint parties)

    cdef bint wait(self)

    cdef void interrupt(self)


@cython.final
cdef class TPrivRef:
    cdef public object v

    @staticmethod
    cdef TPrivRef new()
