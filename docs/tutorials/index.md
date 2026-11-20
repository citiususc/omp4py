# Tutorials and Examples

Every program in this section lives in `docs/examples/` and was executed with the pure
runtime on a free-threaded Python 3.14 while this documentation was written. Outputs
shown are real; only the order of lines printed by several threads varies.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Tutorial
  - You will learn
* - {doc}`pi`
  - `parallel for`, `reduction`, `num_threads`, correctness checks
* - {doc}`matmul`
  - `collapse`, nested loops, writing disjoint output
* - {doc}`histogram`
  - `critical`, private partial results, `nowait`
* - {doc}`numpy`
  - Combining OMP4Py with libraries that release the GIL
```

Also in the repository, `examples/omp4py_examples` contains larger benchmark programs
(FFT, Jacobi, LU decomposition, molecular dynamics, graph clustering, ...). They are
described in {doc}`/benchmarks`.

```{admonition} About scalability
:class: important

The tutorials show **correct** parallel programs. How well they *scale* depends on the Python
interpreter. Threads can only run Python code at the same time on a **free-threaded** build, and
free-threading is still maturing: on Python 3.13 code with many Python objects may work
correctly but scale poorly, on 3.14 it improves, and later versions are expected to keep
improving. If a tutorial does not speed up on your machine, check your interpreter first
({doc}`/user_guide/performance`); the limit is not in the directives.
```

## Running the examples

From the root of the repository:

```bash
pip install -e .
python docs/examples/pi.py
```

To force the pure runtime:

```bash
OMP4PY_PURE=1 python docs/examples/pi.py          # Linux / macOS
$env:OMP4PY_PURE = "1"; python docs/examples/pi.py # PowerShell
```
