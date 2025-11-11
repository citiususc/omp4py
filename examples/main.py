import sys

from omputils import set_omp_threads, set_omp_mode


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
        import fft
        fft.fft(*args)
    elif test == "graphc":
        import graphc
        graphc.graphc(*args)
    elif test == "fib":
        import fib
        fib.fib(*args)
    elif test == "jacobi":
        import jacobi
        jacobi.jacobi(*args)
    elif test == "lud":
        import lud
        lud.lud(*args)
    elif test == "maze":
        import maze
        maze.maze(*args)
    elif test == "md":
        import md
        md.md(*args)
    elif test == "pi":
        import pi
        pi.pi(*args)
    elif test == "qsort":
        import qsort
        qsort.qsort(*args)
    elif test == "quad":
        import quad
        quad.quad(*args)
    elif test == "wordcount":
        import wordcount
        wordcount.wordcount(*args)


if __name__ == '__main__':
    main()
