from omp4py import omp

omp(extension="profile_ext")


@omp(device="profile", parsers=["timed"])
def run(n):
    total = 0
    with timed("whole function"):  # noqa: F821
        with omp("parallel for reduction(+:total) num_threads(4)"):
            for i in range(n):
                total += i
    return total


print(run(200_000))
