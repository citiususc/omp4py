"""Compiled atomic synchronization primitives used by the `omp4py` runtime.

This module provides the Cython implementation of the atomic primitives
defined in the Python runtime. It uses the C11 `<stdatomic.h>` API to
implement lock-free atomic operations for improved performance.

The exposed classes (`AtomicFlag`, `AtomicInt`, and `AtomicObject`)
provide the same behavior and interface as their pure Python counterparts,
but internally rely on low-level atomic instructions instead of Python
locks.

These implementations are used when `omp4py` is compiled with Cython,
allowing the runtime to perform atomic operations efficiently while
keeping the same API as the pure Python runtime.
"""

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_XINCREF, Py_DECREF, Py_XDECREF
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
    uintptr_t atomic_exchange_up "atomic_exchange"(atomic_uintptr_t *obj, uintptr_t desired)
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
        cdef AtomicFlag obj  = AtomicFlag.__new__(AtomicFlag)
        obj._value = atomic_flag_init()
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef bint test_and_set(self):
        return atomic_flag_test_and_set(&self._value)

    cdef void clear(self):
        atomic_flag_clear(&self._value)

cdef class AtomicInt:
    @staticmethod
    cdef AtomicInt new(pyint value = 0):
        cdef AtomicInt obj = AtomicInt.__new__(AtomicInt)
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

    cdef pyint fetch_add(self, pyint arg):
        return atomic_fetch_add_ll(&self._value, arg)

    cdef pyint sub(self, pyint arg):
        return atomic_fetch_sub_ll(&self._value, arg) - arg

    cdef pyint fetch_sub(self, pyint arg):
        return atomic_fetch_sub_ll(&self._value, arg)

    cdef pyint or_(self, pyint arg):
        return atomic_fetch_or_ll(&self._value, arg) | arg

    cdef pyint fetch_or(self, pyint arg):
        return atomic_fetch_or_ll(&self._value, arg)

    cdef pyint xor(self, pyint arg):
        return atomic_fetch_xor_ll(&self._value, arg) ^ arg

    cdef pyint fetch_xor(self, pyint arg):
        return atomic_fetch_xor_ll(&self._value, arg)

    cdef pyint and_(self, pyint arg):
        return atomic_fetch_and_ll(&self._value, arg) & arg

    cdef pyint fetch_and(self, pyint arg):
        return atomic_fetch_and_ll(&self._value, arg)


cdef class AtomicObject:
    @staticmethod
    cdef AtomicObject new():
        cdef AtomicObject obj = AtomicObject.__new__(AtomicObject)
        cdef PyObject * none = <PyObject *> None
        atomic_store_up(&obj._value, <uintptr_t> none)
        Py_XINCREF(none)
        return obj

    def __init__(self):
        raise ValueError("use new() class method")

    cdef object get(self):
        return <object> <PyObject *> atomic_load_up(&self._value)

    cdef bint set(self, object value):
        cdef PyObject * none = <PyObject *> None
        if atomic_compare_exchange_strong_up(&self._value, <uintptr_t *> &none, <uintptr_t> <PyObject *> value):
            Py_XDECREF(none)
            Py_INCREF(value)
            return True
        return False

    cdef object exchange(self, object desired):
        cdef object old = <object> <PyObject *> (
            atomic_exchange_up(&self._value, <uintptr_t> <PyObject *> desired))
        Py_DECREF(self._value)
        return old

    cdef bint compare_exchange(self, object expected, object desired):
        cdef PyObject * c_expected = <PyObject *> expected
        if atomic_compare_exchange_strong_up(&self._value, <uintptr_t *> &c_expected, <uintptr_t> <PyObject *> desired):
            Py_DECREF(expected)
            Py_INCREF(desired)
            return True
        return False

    def __dealloc__(self):
        Py_XDECREF(<PyObject *>self._value)