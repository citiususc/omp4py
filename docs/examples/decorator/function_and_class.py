from omp4py import omp


@omp
def total(n):
    s = 0
    with omp("parallel for reduction(+:s) num_threads(2)"):
        for i in range(n):
            s += i
    return s


@omp
class Solver:
    def __init__(self, n):
        self.n = n

    def run(self):
        s = 0
        with omp("parallel for reduction(+:s) num_threads(2)"):
            for i in range(self.n):
                s += i
        return s


# omp can also be called as a function on an existing object:
def plain(n):
    s = 0
    with omp("parallel for reduction(+:s) num_threads(2)"):
        for i in range(n):
            s += i
    return s


print(total(10), Solver(10).run(), omp(plain)(10))
