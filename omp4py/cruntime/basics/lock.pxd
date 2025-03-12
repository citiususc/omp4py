from omp4py.cruntime.basics.types cimport *

cdef class Mutex:
    cdef object _lock

    @staticmethod
    cdef Mutex new()

    cdef void lock(self)

    cdef void unlock(self)

    cdef bint test(self)


cdef class RMutex:
    cdef object _lock

    @staticmethod
    cdef RMutex new()

    cdef void lock(self)

    cdef void unlock(self)

    cdef bint test(self)


cdef class Barrier:
    cdef object _value

    @staticmethod
    cdef Barrier new(pyint parties)

    cdef void wait(self)
