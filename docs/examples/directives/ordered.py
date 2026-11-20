from omp4py import *


@omp
def ordered_output(n):
    out = []
    with omp("parallel for ordered schedule(dynamic) num_threads(4)"):
        for i in range(n):
            square = i * i             # runs in parallel
            with omp("ordered"):
                out.append(square)     # runs in iteration order
    return out


print(ordered_output(8))
