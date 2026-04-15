from omp4py.runtime.icvs cimport Data
from omp4py.runtime.lowlevel.numeric cimport pyint
from omp4py.runtime.tasks.context cimport TaskContext
from omp4py.runtime.tasks.task cimport Barrier, SharedContext, Task

cdef class ParallelTask(Task):
    cdef object init
    cdef object f

    @staticmethod
    cdef ParallelTask new(object init, object f, SharedContext shared, Data icvs, Barrier barrier)


cdef void set_nthreads(TaskContext ctx, Data icvs, bint active, tuple num_threads)

cpdef void parallel(object f , bint enable, tuple num_threads, pyint proc_bind, tuple copyin)

cpdef void _parallel_thread_init(TaskContext ctx)

cdef void _parallel_main()