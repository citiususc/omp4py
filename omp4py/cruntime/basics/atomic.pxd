from omp4py.cruntime.basics.types cimport *
from libc.stdint cimport uintptr_t

cdef extern from "<stdatomic.h>":
    """
    #define p_atomic_load atomic_load
    #define p_atomic_store atomic_store
    #define p_atomic_compare_exchange_strong atomic_compare_exchange_strong

    atomic_flag atomic_flag_init(){atomic_flag v = ATOMIC_FLAG_INIT; return v;}
    """
    ctypedef struct atomic_flag:
        pass
    cdef atomic_flag atomic_flag_init()
    cdef bint atomic_flag_test_and_set(atomic_flag *flag)
    cdef void atomic_flag_clear(atomic_flag *flag)

    ctypedef struct atomic_uintptr_t:
        pass
    cdef uintptr_t p_atomic_load(atomic_uintptr_t *obj)
    cdef void p_atomic_store(atomic_uintptr_t *obj, uintptr_t desired)
    cdef bint p_atomic_compare_exchange_strong(atomic_uintptr_t *obj, uintptr_t * expected, uintptr_t desired)

    ctypedef struct atomic_llong:
        pass
    cdef pyint atomic_load(atomic_llong *obj)
    cdef void atomic_store(atomic_llong *obj, pyint desired)
    cdef pyint atomic_exchange(atomic_llong *obj, pyint desired)
    cdef bint atomic_compare_exchange_strong(atomic_llong *obj, pyint *expected, pyint desired)
    cdef bint atomic_compare_exchange_weak(atomic_llong *obj, pyint *expected, pyint desired)
    cdef pyint atomic_fetch_add(atomic_llong *obj, pyint arg)
    cdef pyint atomic_fetch_sub(atomic_llong *obj, pyint arg)
    cdef pyint atomic_fetch_or(atomic_llong *obj, pyint arg)
    cdef pyint atomic_fetch_xor(atomic_llong *obj, pyint arg)
    cdef pyint atomic_fetch_and(atomic_llong *obj, pyint arg)


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
