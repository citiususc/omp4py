# Benchmarks

The folder `examples/` of the repository is a separate small package
(`omp4py_examples`) with the programs used for the performance evaluation in the
[papers](index.md). It installs a command, `omp4pyex`:

```bash
omp4pyex <mode> <test> <threads> [args...]
```

| Argument | Values |
| --- | --- |
| `mode` | `0` pure, `1` hybrid, `2` compiled, `3` compiled with types, `-1` PyOMP (Numba) for comparison |
| `test` | `fft`, `graphc`, `fib`, `jacobi`, `lud`, `maze`, `md`, `pi`, `qsort`, `quad`, `wordcount` |
| `threads` | Number of threads |
| `args` | Positional arguments of the benchmark (`n`, `seed`, ...); each is `eval`-ed |

With `uv`:

```bash
cd examples
uv run -p 3.14t omp4pyex 1 pi 4 2000000
```

## The programs

| Benchmark | What it computes | Default size | Directives it uses |
| --- | --- | --- | --- |
| `fft` | Fast Fourier Transform of a complex vector | 2²⁴ numbers | `parallel`, `for` |
| `graphc` | Clustering coefficient of a NetworkX graph | 300k vertices, 100 edges each | `parallel for` (non-numeric objects; PyOMP cannot run it) |
| `jacobi` | Jacobi iteration for `A·x = b` | 3000 × 3000, ≤ 1000 iterations | `parallel`, `single`, `for reduction`, `barrier` (also with `mpi4py`) |
| `lud` | LU decomposition | 2000 × 2000 | `parallel`, `single`, `for` |
| `maze` | Breadth-first search on a grid | `n=10` | `parallel`, `single`, **`task firstprivate`** |
| `md` | Molecular dynamics | 8000 particles, 10 steps | `parallel for`, `parallel reduction`, `for` |
| `pi` | Riemann integral for π | n = 2·10⁹ | `parallel for reduction` |
| `qsort` | Quicksort | 4·10⁸ elements, limit 10⁵ | `parallel`, `single`, **`task if`** |
| `quad` | Numerical quadrature | | `parallel for reduction` |
| `fib` | Recursive Fibonacci | | `parallel`, `single`, **`task if`**, **`taskwait`** |
| `wordcount` | Word frequency | 10⁶ words or a file | `parallel`, `for`, `critical` |

The default sizes are those of the paper (HPC scale); use smaller ones to try them on a laptop.

## Pi in the four modes

`examples/omp4py_examples/pi.py` contains the same kernel for every mode. Compare them:

```{literalinclude} ../examples/omp4py_examples/pi.py
:language: python
:lines: 14-35
```

The first function is the PyOMP/Numba version; the other two are the OMP4Py versions. The
second is selected by `omp(compile=use_compiled())`, the third adds type annotations (mode 3).

## Measuring your own code

1. Warm up before timing ({doc}`/user_guide/performance`).
2. Run with 1, 2, 4, ... threads (`omp_set_num_threads` or `OMP_NUM_THREADS`).
3. Report speedup `T(1)/T(p)` and efficiency `speedup/p`.
4. Always state the mode, the Python build (`sys._is_gil_enabled()`), and the hardware.

This repository does not publish benchmark numbers in the documentation; see the papers cited on
the {doc}`home page <index>` for the measured results.
