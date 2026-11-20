from omp4py import *


@omp
def broadcast():
    seen = []
    token = None
    with omp("parallel num_threads(4)"):
        with omp("single copyprivate(token)"):
            token = "computed once"
        with omp("critical"):
            seen.append(token)
    return seen


print(broadcast())
