from omp4py import *

omp_set_num_threads(3)
print("max threads:", omp_get_max_threads())
print("inside a region?", omp_in_parallel())


@omp
def inside():
    with omp("parallel"):
        with omp("single"):
            print("team size:", omp_get_num_threads(), "| inside a region?", omp_in_parallel())


inside()

omp_set_schedule(omp_sched_dynamic, 8)
print("schedule(runtime) is now", omp_get_schedule())
