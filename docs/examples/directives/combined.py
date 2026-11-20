from omp4py import *


@omp
def combined(n):
    out = []
    total = 0

    # worksharing loop combined with its parallel region
    with omp("parallel for reduction(+:total) num_threads(3)"):
        for i in range(n):
            total += i

    # OpenMP 6.0 name: masked (master is deprecated and cannot be combined)
    with omp("parallel masked num_threads(3)"):
        out.append("masked: only thread 0")

    with omp("parallel single num_threads(3)"):
        out.append("single: exactly one thread")

    # the directive-name modifier sends `if` to the parallel construct only
    with omp("parallel for if(parallel: n > 100) num_threads(3) schedule(monotonic: dynamic, 2)"):
        for i in range(4):
            pass

    return total, out


print(combined(10))
