# Performance

```{contents}
:local:
:depth: 2
```

## OMP4Py divides the work; Python computes it

This is the most important idea of the whole page.

OMP4Py's job is to **split the work** among threads and to coordinate them (scheduling,
barriers, locks). It does **not** make the computation of each thread faster: the work inside a
parallel region is run by the Python interpreter (or by the code Cython generates in the
compiled modes), exactly as it would be in a sequential program.

Look at what the simplest worksharing loop becomes:

```python
with omp("parallel for schedule(static)"):
    for i in range(n):
        x = (i * 7 + 3) % 11
        total += x * x
```

```python
# generated (simplified): the scheduler hands each thread one block...
_omp.for_init(bounds, ...)
while _omp.for_next_static(bounds):
    # ...and from here on it is plain Python, until the block is done
    for i in range(bounds.init, bounds.end):
        x = (i * 7 + 3) % 11
        total += x * x
```

With `static` the division is done once at the beginning. After that **all the time is spent in
the Python interpreter** running ordinary bytecode in several threads at once. Whether those threads
really advance in parallel depends on the interpreter, not on OMP4Py.

### Scalability depends on the interpreter

Free-threaded Python is young. Even without the GIL, threads that execute Python code contend
for shared interpreter resources (reference counting of shared objects, memory allocation,
internal locks, ...). The interpreter authors are reducing this contention release after
release, so the same OMP4Py program can scale differently on different Python versions:

- On **3.13t** code that works with many Python objects can run correctly but scale poorly.
- On **3.14t** it improves, but some workloads still show limited scaling.
- Newer versions keep improving it, and OMP4Py benefits **without any change**.

In the OMP4Py paper, the pure-Python implementation reached a maximum speedup of 3.6× (Jacobi)
on a 3.14 beta interpreter, and the authors report that results on 3.13 are worse. Scalability
limitations of that kind come from the interpreter and are expected to fade as free-threading
matures.

You can see how a loop of your own behaves with the program below (a pure-Python loop on a
free-threaded build; run it on your machine):

```{literalinclude} ../examples/scaling.py
:language: python
```

On one test machine (free-threaded 3.14, pure mode) it printed:

```text
1 threads: 0.290s
2 threads: 0.142s
4 threads: 0.110s
8 threads: 0.060s
```

Your numbers will differ with the machine, the Python build and the work done in the loop. If
your own measurement does not scale, first check the Python version and the kind of work (object
allocation, shared containers) before suspecting the directives.

```{tip}
When you report or compare performance, state the Python version and whether it is
free-threaded, the OMP4Py {doc}`mode <execution_modes>`, and the number of threads.
```

## The GIL

A normal Python build has the **Global Interpreter Lock**: only one thread executes Python
bytecode at a time. With it:

- OMP4Py programs are **correct**, but CPU-bound parallel regions do **not** run faster: the
  threads take turns.
- You still gain when the threads spend their time **waiting** (I/O, sleeping) or inside native
  libraries that release the GIL.
- Check your interpreter: `python -c "import sys; print(sys._is_gil_enabled())"`. You want
  `False`.

For CPU-bound parallelism use a **free-threaded** build (`3.13t`, `3.14t`, ...).

```{note}
Once device offloading (`target`) is available in OMP4Py, the computation of the offloaded
regions will not depend on the interpreter running the host program, so the GIL will stop being
a limit for programs whose goal is to use that device.
```

## What you can do about it

In decreasing order of impact:

1. **Use a free-threaded Python**, the newest you can.
2. **Move the work out of the interpreter**: use {doc}`compiled mode <compiled_mode>` with type
   annotations for numeric loops, or call libraries that run native code (NumPy, ...).
3. **Reduce shared mutable state** between threads (see below).
4. **Balance the work** with the right schedule.

## OpenMP good practices

### One parallel region, many loops

Entering a `parallel` region creates the team and joining it ends it. Do it once and put several
loops inside:

```python
# slow: a team is created and joined on every iteration of the outer loop
for step in range(1000):
    with omp("parallel for"):
        for i in range(n): ...

# better: the team lives across the iterations
with omp("parallel"):
    for step in range(1000):
        with omp("for"):
            for i in range(n): ...
```

### Remove barriers you do not need

`for`, `sections` and `single` end with an implicit barrier. If the next loop does not depend
on the results of the previous one, add `nowait`:

```python
with omp("parallel"):
    with omp("for nowait"):
        for i in range(n): a[i] = f(i)
    with omp("for"):
        for i in range(n): b[i] = g(i)       # independent of a
```

### Do not parallelise tiny work

A parallel region has a start-up cost. Use `if` to run sequentially for small inputs:
`parallel for if(n > 10_000)`.

### Choose the schedule for the work

| Work per iteration | Schedule |
| --- | --- |
| Equal | `static` (default): lowest overhead |
| Very different | `dynamic, chunk` with a chunk of 8-64 |
| Decreasing or increasing | `guided` |

Larger chunks mean less scheduling overhead but worse balance. Try `schedule(runtime)` together
with `OMP_SCHEDULE` to experiment without changing the code.

### Avoid contention

- A `critical` section serialises the threads that reach it. Prefer a `reduction`, or **private
  partial results merged once** at the end ({doc}`/tutorials/histogram`).
- Do not make all threads update one shared container in the hot loop.
- Give each thread its own contiguous part of an output (`schedule(static)` does this by
  default) instead of interleaving writes from several threads.
- Keep thread-private data in private variables; they are cheaper than shared ones.

#### Variables of your function inside a region

The block of a `parallel` and of a `task` is turned into a **function** nested in yours, so the
variables of your function that the block uses become **closure variables** (*cells*) shared by all the
threads. In the current implementations of the free-threaded interpreter, reading one of them in every
iteration of a loop, from many threads at once, can reduce the performance a lot: the loop can even run
slower with eight threads than with one.

```python
w = 1.0 / n                                 # a variable of the function
with omp("parallel for reduction(+:total)"):
    for i in range(n):
        x = (i + 0.5) * w                   # every thread reads `w` in every iteration
```

If the threads only **read** the variable, make it `firstprivate`: every thread gets its own local name
for the value, which is cheap to read.

```python
with omp("parallel for reduction(+:total) firstprivate(w)"):
    ...
```

To avoid writing `firstprivate` for every variable, enable the `omp_local_aliases` modifier. In every
function nested in another one (the block of a `parallel` is one), it gives a local name to each variable
that the function only reads, which has the same effect as `firstprivate`. The variables have to be
initialised when the function starts, so it does not allow a variable that is only assigned in some
branches of the code before the region:

```python
@omp(modifiers={"omp_local_aliases": True})
def pi(n):
    ...
```

If the variable has to be **shared** because the threads update it, avoid rebinding it inside the loop:
keep the value inside a shared object (a list, a dictionary or an instance) and change its content, or
give each thread its own copy and combine them at the end with a `reduction`.

### Keep the parallel work coarse

Parallelise the **outermost loop with enough iterations** (use `collapse` to enlarge the space),
and make each iteration do a meaningful amount of work. Many tiny iterations spend more time in
scheduling than in computing.

### Measure properly

```python
import time

kernel(small_input)                        # warm-up: preprocessing and compilation happen here
t0 = time.perf_counter()
result = kernel(data)
print(time.perf_counter() - t0)
```

- The first call of a compiled function pays the build cost.
- Vary the number of threads (`omp_set_num_threads` or `OMP_NUM_THREADS`) and compute speedup
  `T(1)/T(p)` and efficiency `speedup/p`.
- See {doc}`/benchmarks` for the benchmark programs of the repository.
