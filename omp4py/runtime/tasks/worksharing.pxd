from libc cimport math # replace python module

from omp4py.runtime.lowlevel.atomic cimport AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.mutex cimport Event
from omp4py.runtime.lowlevel.numeric cimport pyint, pyint_array
from omp4py.runtime.tasks.context cimport TaskContext
from omp4py.runtime.tasks.task cimport Task


cdef const pyint _static
cdef const pyint _dynamic
cdef const pyint _guided
cdef const pyint _runtime


# bounds header ranges
cdef const pyint _r_start
cdef const pyint _r_stop
cdef const pyint _r_step
cdef const pyint _r_mod
cdef const pyint _r_div
cdef const pyint _r_off
cdef const pyint _rs_len


cdef class ForShared:
    cdef AtomicInt count 

    @staticmethod
    cdef ForShared new(pyint count)

cdef class OrderedShared:
    cdef pyint count
    cdef pyint it
    cdef pyint own
    cdef AtomicObject event

    @staticmethod
    cdef OrderedShared new(pyint count)


cdef class ForBounds:
    cdef readonly pyint init
    cdef readonly pyint end
    cdef public it
    cdef pyint step
    cdef pyint count
    cdef pyint chunk
    cdef pyint its
    cdef pyint collapse
    cdef readonly pyint_array rs


ctypedef bint (*kind_t)(ForBounds)

cdef class ForTask(Task):
    cdef ForBounds bounds
    cdef kind_t kind
    cdef AtomicInt shared_count
    cdef OrderedShared ordered

    @staticmethod
    cdef ForTask new(TaskContext ctx, ForBounds bounds)

cpdef ForBounds for_bounds(pyint collapse)

cdef void set_bounds(ForBounds bounds)

cpdef void for_init(ForBounds bounds, pyint modifier, pyint schedule, pyint chunk, pyint ordered)

cpdef bint for_next_runtime(ForBounds bounds)

cdef bint for_next_static(ForBounds bounds) # Must be cdef to be used as a function pointer; manually exported.

cdef bint for_next_dynamic(ForBounds bounds)

cdef bint for_next_guided(ForBounds bounds)

cpdef void for_end(bint nowait)

cdef bint fix_bounds(ForBounds bounds)

cpdef void ordered_start()

cpdef void ordered_next()


cdef class SectionsShared:
    cdef AtomicInt count

    @staticmethod
    cdef SectionsShared new(pyint n)


cdef class SectionsTask(Task):
    cdef SectionsShared shared

    @staticmethod
    cdef SectionsTask new(TaskContext ctx, SectionsShared shared)


cpdef void sections_init(pyint sections)

cpdef pyint sections_next()

cpdef void sections_end(bint nowait)


cdef class SingleShared:
    cdef AtomicInt count
    cdef Event event
    cdef SingleCopyPrivate copyprivate

    @staticmethod
    cdef SingleShared new(bint copyprivate)


cdef class SingleTask(Task):
    cdef SingleShared shared

    @staticmethod
    cdef SingleTask new(TaskContext ctx, SingleShared shared)


cpdef bint single_init(bint copyprivate)

cpdef void single_end(bint nowait)


cdef class SingleCopyPrivate:
    cdef public object v0
    cdef public object v1
    cdef public object v2
    cdef public object v3
    cdef public object v4
    cdef public object v5
    cdef public object v6
    cdef public object v7
    cdef public SingleCopyPrivate next

    @staticmethod
    cdef SingleCopyPrivate new(pyint nvars)


cpdef SingleCopyPrivate single_copy_get(pyint nvars)

cpdef void single_copy_notify()

cpdef SingleCopyPrivate single_copy_wait()
