from omp4py import omp

# A decorator factory with defaults. Because the new name differs from "omp",
# the alias must be given so that directives written with it are recognised.
fast = omp(alias="fast", compile=False)


@fast
def kernel(n):
    total = 0
    with fast("parallel for reduction(+:total) num_threads(2)"):   # directive written with the alias
        for i in range(n):
            total += i
    return total


print(kernel(10))
