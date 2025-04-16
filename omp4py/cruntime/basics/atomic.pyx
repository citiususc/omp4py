from cpython.object cimport PyObject

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
        p_atomic_store(&obj._value, 0)
        obj._holder = None
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef object get(self):
        ref: uintptr_t = p_atomic_load(&self._value)
        if ref == 0:
            return None
        return <object> <PyObject *> ref

    cdef bint set(self, object value):
        cdef uintptr_t ZERO = <uintptr_t> 0
        if p_atomic_compare_exchange_strong(&self._value, &ZERO, <uintptr_t> <PyObject *> value):
            self._holder = <object> <PyObject *> p_atomic_load(&self._value)
            return True
        return False

cdef class AtomicInt:
    @staticmethod
    cdef AtomicInt new(pyint value = 0):
        obj: AtomicInt = AtomicInt.__new__(AtomicInt)
        atomic_store(&obj._value, value)
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef void set(self, pyint value):
        atomic_store(&self._value, value)

    cdef pyint get(self):
        return atomic_load(&self._value)

    cdef pyint exchange(self, pyint desired):
        return atomic_exchange(&self._value, desired)

    cdef bint compare_exchange_strong(self, pyint expected, pyint desired):
        return atomic_compare_exchange_strong(&self._value, &expected, desired)

    cdef bint compare_exchange_weak(self, pyint expected, pyint desired):
        return atomic_compare_exchange_weak(&self._value, &expected, desired)

    cdef pyint add(self, pyint arg):
        return atomic_fetch_add(&self._value, arg) + arg

    cdef pyint sub(self, pyint arg):
        return atomic_fetch_sub(&self._value, arg) - arg

    cdef pyint or_(self, pyint arg):
        return atomic_fetch_or(&self._value, arg) | arg

    cdef pyint xor(self, pyint arg):
        return atomic_fetch_xor(&self._value, arg) ^ arg

    cdef pyint and_(self, pyint arg):
        return atomic_fetch_and(&self._value, arg) & arg
