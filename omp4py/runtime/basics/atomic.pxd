from omp4py.runtime.basics.types cimport *

cdef extern from "<stdatomic.h>":
    ctypedef struct atomic_flag:
        pass

    ctypedef struct atomic_uintptr_t:
        pass

    ctypedef struct atomic_llong:
        pass


cdef class AtomicFlag:
    cdef atomic_flag _value

    @staticmethod
    cdef AtomicFlag new()

    cdef bint no_clear_test_and_set(self)

    cdef bint test_and_set(self)

    cdef void clear(self)

cdef class AtomicObject:
    cdef object _holder
    cdef atomic_uintptr_t _value

    @staticmethod
    cdef AtomicObject new()

    cdef object get(self)

    cdef bint set(self, object value)

cdef class AtomicInt:
    cdef atomic_llong _value

    @staticmethod
    cdef AtomicInt new(pyint value= *)

    cdef void set(self, pyint value)

    cdef pyint get(self)

    cdef pyint exchange(self, pyint desired)

    cdef bint compare_exchange_strong(self, pyint expected, pyint desired)

    cdef bint compare_exchange_weak(self, pyint expected, pyint desired)

    cdef pyint add(self, pyint arg)

    cdef pyint sub(self, pyint arg)

    cdef pyint or_(self, pyint arg)

    cdef pyint xor(self, pyint arg)

    cdef pyint and_(self, pyint arg)
