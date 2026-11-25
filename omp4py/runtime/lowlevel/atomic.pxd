cimport cython
from omp4py.runtime.lowlevel.numeric cimport pyint

cdef extern from "<stdatomic.h>":
    ctypedef struct atomic_flag:
        pass

    ctypedef struct atomic_uintptr_t:
        pass

    ctypedef struct atomic_llong:
        pass


@cython.final
cdef class AtomicFlag:
    cdef atomic_flag _value

    @staticmethod
    cdef AtomicFlag new()

    cdef bint test_and_set(self) noexcept

    cdef void clear(self) noexcept


@cython.final
cdef class AtomicObject:
    cdef atomic_uintptr_t _value
    cdef atomic_llong _readers

    @staticmethod
    cdef AtomicObject new()

    cdef object get(self)

    cdef bint set(self, object value) noexcept

    cdef object exchange(self, object desired)

    cdef bint compare_exchange(self, object expected, object desired) noexcept

@cython.final
cdef class AtomicInt:
    cdef atomic_llong _value

    @staticmethod
    cdef AtomicInt new(pyint value= *)

    cdef void set(self, pyint value) noexcept

    cdef pyint get(self) noexcept

    cdef pyint exchange(self, pyint desired) noexcept

    cdef bint compare_exchange_strong(self, pyint expected, pyint desired) noexcept

    cdef bint compare_exchange_weak(self, pyint expected, pyint desired) noexcept

    cdef pyint add(self, pyint arg) noexcept

    cdef pyint fetch_add(self, pyint arg) noexcept

    cdef pyint sub(self, pyint arg) noexcept

    cdef pyint fetch_sub(self, pyint arg) noexcept

    cdef pyint or_(self, pyint arg) noexcept

    cdef pyint fetch_or(self, pyint arg) noexcept

    cdef pyint xor(self, pyint arg) noexcept

    cdef pyint fetch_xor(self, pyint arg) noexcept

    cdef pyint and_(self, pyint arg) noexcept

    cdef pyint fetch_and(self, pyint arg) noexcept
