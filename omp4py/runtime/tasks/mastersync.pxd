from omp4py.runtime.lowlevel.numeric cimport pyint
from omp4py.runtime.lowlevel.mutex cimport Mutex

cdef dict[str, Mutex] _critical_ids
cdef Mutex _critical_mutex
cdef Mutex _critical_unnamed

cpdef void barrier()

cpdef bint masked(pyint select)

cpdef Mutex critical_named(str name)

cpdef void critical_named_lock(Mutex m)

cpdef void critical_named_unlock(Mutex m)

cpdef void critical_lock()

cpdef void critical_unlock()

cpdef void team_lock()

cpdef void team_unlock()
