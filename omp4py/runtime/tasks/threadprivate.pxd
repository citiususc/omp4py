from omp4py.runtime.lowlevel.mutex cimport Mutex
from omp4py.runtime.lowlevel.numeric cimport pyint, pyint_array
from omp4py.runtime.tasks.context cimport  omp_ctx
from omp4py.runtime.tasks.task cimport TPrivRef
from cpython.list cimport PyList_GET_ITEM

cdef pyint _threadprivate_ids_last
cdef Mutex _threadprivate_ids_mutex


cpdef pyint threadprivate(str name, object value)

cpdef inline TPrivRef threadprivates(pyint i):
    tpvars = omp_ctx().tpvars
    if i >= len(tpvars):
        update_privates(tpvars)
    return <object> PyList_GET_ITEM(tpvars, i) # Fast version of tpvars[i]

cdef void update_privates(list[TPrivRef] tpvars)

cdef pyint_array map_privates(tuple names)

cdef object copy_private(object obj)
