from cpython.object cimport PyObject
from libc.stdint cimport uintptr_t

cdef extern from "<stdatomic.h>":
    """
    static inline atomic_flag atomic_flag_init(){ atomic_flag v = ATOMIC_FLAG_INIT; return v;}
    """

    atomic_flag atomic_flag_init()
    bint atomic_flag_test_and_set(atomic_flag *flag)
    void atomic_flag_clear(atomic_flag *flag)

    uintptr_t atomic_load_up "atomic_load"(atomic_uintptr_t *obj)
    void atomic_store_up "atomic_store"(atomic_uintptr_t *obj, uintptr_t desired)
    bint atomic_compare_exchange_strong_up "atomic_compare_exchange_strong"(atomic_uintptr_t *obj, uintptr_t * expected,
                                                                            uintptr_t desired)

    pyint atomic_load_ll "atomic_load"(atomic_llong *obj)
    void atomic_store_ll"atomic_store"(atomic_llong *obj, pyint desired)
    pyint atomic_exchange_ll"atomic_exchange"(atomic_llong *obj, pyint desired)
    pyint atomic_fetch_add_ll"atomic_fetch_add"(atomic_llong *obj, pyint arg)
    pyint atomic_fetch_sub_ll"atomic_fetch_sub"(atomic_llong *obj, pyint arg)
    pyint atomic_fetch_or_ll"atomic_fetch_or"(atomic_llong *obj, pyint arg)
    pyint atomic_fetch_xor_ll"atomic_fetch_xor"(atomic_llong *obj, pyint arg)
    pyint atomic_fetch_and_ll"atomic_fetch_and"(atomic_llong *obj, pyint arg)
    bint atomic_compare_exchange_strong_ll "atomic_compare_exchange_strong"(atomic_llong *obj, pyint *expected,
                                                                            pyint desired)
    bint atomic_compare_exchange_weak_ll "atomic_compare_exchange_weak"(atomic_llong *obj, pyint *expected,
                                                                        pyint desired)

cdef class AtomicFlag:
    @staticmethod
    cdef AtomicFlag new():
        obj: AtomicFlag = AtomicFlag.__new__(AtomicFlag)
        obj._value = atomic_flag_init()
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef bint no_clear_test_and_set(self):
        return atomic_flag_test_and_set(&self._value)

    cdef bint test_and_set(self):
        return atomic_flag_test_and_set(&self._value)

    cdef void clear(self):
        atomic_flag_clear(&self._value)

cdef class AtomicObject:
    @staticmethod
    cdef AtomicObject new():
        obj: AtomicObject = AtomicObject.__new__(AtomicObject)
        atomic_store_up(&obj._value, 0)
        obj._holder = None
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef object get(self):
        ref: uintptr_t = atomic_load_up(&self._value)
        if ref == 0:
            return None
        return <object> <PyObject *> ref

    cdef bint set(self, object value):
        cdef uintptr_t ZERO = <uintptr_t> 0
        if atomic_compare_exchange_strong_up(&self._value, &ZERO, <uintptr_t> <PyObject *> value):
            self._holder = <object> <PyObject *> atomic_load_up(&self._value)
            return True
        return False

cdef class AtomicInt:
    @staticmethod
    cdef AtomicInt new(pyint value = 0):
        obj: AtomicInt = AtomicInt.__new__(AtomicInt)
        atomic_store_ll(&obj._value, value)
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef void set(self, pyint value):
        atomic_store_ll(&self._value, value)

    cdef pyint get(self):
        return atomic_load_ll(&self._value)

    cdef pyint exchange(self, pyint desired):
        return atomic_exchange_ll(&self._value, desired)

    cdef bint compare_exchange_strong(self, pyint expected, pyint desired):
        return atomic_compare_exchange_strong_ll(&self._value, &expected, desired)

    cdef bint compare_exchange_weak(self, pyint expected, pyint desired):
        return atomic_compare_exchange_weak_ll(&self._value, &expected, desired)

    cdef pyint add(self, pyint arg):
        return atomic_fetch_add_ll(&self._value, arg) + arg

    cdef pyint sub(self, pyint arg):
        return atomic_fetch_sub_ll(&self._value, arg) - arg

    cdef pyint or_(self, pyint arg):
        return atomic_fetch_or_ll(&self._value, arg) | arg

    cdef pyint xor(self, pyint arg):
        return atomic_fetch_xor_ll(&self._value, arg) ^ arg

    cdef pyint and_(self, pyint arg):
        return atomic_fetch_and_ll(&self._value, arg) & arg
