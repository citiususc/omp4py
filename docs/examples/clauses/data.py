from omp4py import *


@omp
def data_sharing():
    shared_value = 0
    private_value = 10
    first = 5
    last = -1

    with omp("parallel num_threads(4) shared(shared_value) private(private_value) firstprivate(first)"):
        private_value = omp_get_thread_num()     # each thread has its own copy
        first += 1                               # every copy starts at 5
        with omp("critical"):
            shared_value += 1

    with omp("parallel for lastprivate(last) num_threads(4)"):
        for i in range(10):
            last = i

    return shared_value, private_value, first, last


print(data_sharing())
