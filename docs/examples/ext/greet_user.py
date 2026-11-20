from omp4py import omp

omp(extension="greet_ext")          # registered now, imported later (only if the preprocessor runs)


@omp(parsers=["greet"])             # enable the subparser for this function
def main():
    greet("world")                  # noqa: F821  (replaced during preprocessing)
    greet("OMP4Py")                 # noqa: F821


main()
