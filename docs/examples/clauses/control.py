from omp4py import *


@omp
def control(n):
    with omp("parallel if(n > 100) num_threads(4)"):
        with omp("single"):
            print("n =", n, "-> team of", omp_get_num_threads())


control(10)
control(1000)
