import sys

from omputils import set_omp_threads


def main():
    if len(sys.argv) < 3 or (len(sys.argv) < 4 and sys.argv[1] == "numba"):
        print("required: [numba] <test> <threads> [args...]", file=sys.stderr)
        print("available tests: fft, graphc, jacobi, lud, md, pi, qsort, wordcount", file=sys.stderr)
        exit(-1)
    numba = sys.argv[1] == "numba"
    test = sys.argv[1] if not numba else sys.argv[2]
    try:
        threads = int(sys.argv[2] if not numba else sys.argv[3])
    except ValueError:
        print("threads must be a number", file=sys.stderr)
        exit(-1)
    args = list(map(eval, sys.argv[3:] if not numba else sys.argv[4:]))

    set_omp_threads(threads)

    if test == "fft":
        import fft
        fft.fft(*args, numba=numba)
    elif test == "graphc":
        import graphc
        graphc.graphc(*args, numba=numba)
    elif test == "jacobi":
        import jacobi
        jacobi.jacobi(*args, numba=numba)
    elif test == "lud":
        import lud
        lud.lud(*args, numba=numba)
    elif test == "md":
        import md
        md.md(*args, numba=numba)
    elif test == "pi":
        import pi
        pi.pi(*args, numba=numba)
    elif test == "qsort":
        import qsort
        qsort.qsort(*args, numba=numba)
    elif test == "quad":
        import quad
        quad.quad(*args, numba=numba)
    elif test == "wordcount":
        import wordcount
        wordcount.wordcount(numba=numba)


if __name__ == '__main__':
    main()
