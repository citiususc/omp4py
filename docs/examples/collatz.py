from omp4py import *


@omp
def longest_collatz(limit):
    omp("declare reduction(best : tuple) "
        "combiner(omp_out = omp_in if omp_in[0] > omp_out[0] else omp_out) "
        "initializer(omp_priv = (0, 0))")

    best: tuple = (0, 0)               # (steps, starting number)
    with omp("parallel for reduction(best:best) schedule(dynamic, 500) num_threads(4)"):
        for start in range(1, limit):
            n = start
            steps = 0
            while n != 1:
                n = n // 2 if n % 2 == 0 else 3 * n + 1
                steps += 1
            if steps > best[0]:
                best = (steps, start)

    return best


if __name__ == "__main__":
    steps, start = longest_collatz(100_000)
    print(f"below 100000, the longest chain starts at {start} and has {steps} steps")
