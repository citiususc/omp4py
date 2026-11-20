import time

from omp4py import *


@omp
def work(n):
    total = 0
    with omp("parallel for schedule(static) reduction(+:total)"):
        for i in range(n):
            x = (i * 7 + 3) % 11
            total += x * x
    return total


for threads in (1, 2, 4, 8):
    omp_set_num_threads(threads)
    work(1000)                                   # warm-up
    t0 = time.perf_counter()
    work(4_000_000)
    print(f"{threads} threads: {time.perf_counter() - t0:.3f}s")
