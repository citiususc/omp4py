from omp4py.cruntime.basics.types cimport pyint
from omp4py.cruntime.basics cimport array
from omp4py.cruntime.common cimport tasks

cdef pyint _static
cdef pyint _dynamic
cdef pyint _guided
cdef pyint _auto
cdef pyint _runtime

cpdef bint single()

cpdef bint single_end()

cpdef array.iview for_bounds(list[pyint] bounds)

cpdef void for_init(array.iview bounds, pyint kind, pyint chunk, bint monotonic, pyint ordered, pyint order)

cpdef void for_end()

cpdef bint for_next(array.iview bounds)

cdef void for_static(tasks.ForTask task, array.iview bounds)

cdef void for_dynamic(tasks.ForTask task, array.iview bounds)

cdef void for_guided(tasks.ForTask task, array.iview bounds)
