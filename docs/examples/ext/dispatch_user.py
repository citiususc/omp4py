from omp4py import *

omp(extension="dispatch_device")


@omp(device="deterministic")
def owners(n):
    owner = [-1] * n
    with omp("parallel for schedule(dynamic, 1) num_threads(2)"):
        for i in range(n):
            owner[i] = omp_get_thread_num()
    return owner


print(owners(6))
