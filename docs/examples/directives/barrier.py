from omp4py import *


@omp
def phases():
    log = []
    with omp("parallel num_threads(3)"):
        with omp("critical"):
            log.append("phase 1")
        omp("barrier")                 # nobody starts phase 2 before all finished phase 1
        with omp("critical"):
            log.append("phase 2")
    return log


print(phases())
