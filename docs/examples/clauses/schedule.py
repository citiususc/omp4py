from omp4py import *


@omp
def static_chunks(n):
    owner = [-1] * n
    with omp("parallel for schedule(static, 2) num_threads(3)"):
        for i in range(n):
            owner[i] = omp_get_thread_num()
    return owner


@omp
def runtime_schedule(n):
    owner = [-1] * n
    with omp("parallel for schedule(runtime) num_threads(3)"):
        for i in range(n):
            owner[i] = omp_get_thread_num()
    return owner


print("static,2 :", static_chunks(8))
omp_set_schedule(omp_sched_static, 4)
print("runtime  :", runtime_schedule(8))
