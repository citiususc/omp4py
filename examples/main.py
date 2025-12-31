import sys

from omp4py_examples.omputils import set_omp_threads, set_omp_mode

def main():
    if len(sys.argv) < 4:
        print("required: <mode> <test> <threads> [args...]", file=sys.stderr)
        print("available modes: 0 -> pure, 1 -> hybrid, 2 -> compiled, 3 -> compiled with types, -1 -> pyomp",
              file=sys.stderr)
        print("available tests: fft, graphc, fib, jacobi, lud, maze, md, pi, qsort, quad, wordcount", file=sys.stderr)
        exit(-1)
    try:
        mode = int(sys.argv[1])
        if mode < -1 or mode > 3:
            raise ValueError()

        set_omp_mode(mode)
    except ValueError:
        print("mode must be a valid number", file=sys.stderr)
        exit(-1)

    test = sys.argv[2]
    try:
        threads = int(sys.argv[3])
    except ValueError:
        print("threads must be a number", file=sys.stderr)
        exit(-1)
    args = list(map(eval, sys.argv[4:]))

    set_omp_threads(threads)

    if test == "fft":
        import omp4py_examples.fft
        omp4py_examples.fft.fft(*args)
    elif test == "graphc":
        import omp4py_examples.graphc
        omp4py_examples.graphc.graphc(*args)
    elif test == "fib":
        import omp4py_examples.fib
        omp4py_examples.fib.fib(*args)
    elif test == "jacobi":
        import omp4py_examples.jacobi
        omp4py_examples.jacobi.jacobi(*args)
    elif test == "lud":
        import omp4py_examples.lud
        omp4py_examples.lud.lud(*args)
    elif test == "maze":
        import omp4py_examples.maze
        omp4py_examples.maze.maze(*args)
    elif test == "md":
        import omp4py_examples.md
        omp4py_examples.md.md(*args)
    elif test == "pi":
        import omp4py_examples.pi
        omp4py_examples.pi.pi(*args)
    elif test == "qsort":
        import omp4py_examples.qsort
        omp4py_examples.qsort.qsort(*args)
    elif test == "quad":
        import omp4py_examples.quad
        omp4py_examples.quad.quad(*args)
    elif test == "wordcount":
        import omp4py_examples.wordcount
        omp4py_examples.wordcount.wordcount(*args)


if __name__ == '__main__':
    main()
