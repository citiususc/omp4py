from omp4py.runtime.icvs cimport Data
from omp4py.runtime.lowlevel cimport threadlocal # omp4py_threadlocal_get
from omp4py.runtime.tasks.task cimport Task

cdef class TaskContext:
    cdef Data icvs
    cdef Task task
    cdef list tpvars
    cdef list all_tpvars

    @staticmethod
    cdef TaskContext new(Task task, list tpvars)

    cdef void push(self, Task task)

    cdef void pop(self)


cdef class ImplicitTask(Task):

    @staticmethod
    cdef ImplicitTask new()


cdef void context_init()

cdef extern from *:
    # Redefine omp_ctx as a direct call to C function.
    cdef TaskContext omp_ctx "omp4py_threadlocal_get"()
