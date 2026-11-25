cimport cython

from omp4py.runtime.lowlevel.numeric cimport CACHE_LINE, pyint, pyint_array


@cython.final
cdef class Rng:
    cdef pyint_array _state

    @staticmethod
    cdef Rng new(pyint n, pyint seed)

    cdef pyint next(self, pyint tid) noexcept

    cdef pyint victim(self, pyint tid, pyint n) noexcept


cdef inline pyint _slot(pyint tid) noexcept:
    return tid * (CACHE_LINE // sizeof(pyint))  # a cache line between the states of two threads


cdef inline pyint _length(pyint n) noexcept:
    return n * (CACHE_LINE // sizeof(pyint))
