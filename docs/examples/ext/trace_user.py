from omp4py import omp

omp(extension="trace_ext")          # loaded lazily by the preprocessor


@omp(parsers=["trace"])
def work():
    with trace("compute"):  # noqa: F821  (never executed: replaced at preprocessing time)
        total = sum(range(10))
    return total


print(work())
