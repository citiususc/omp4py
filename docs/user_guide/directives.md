# Directives

A **directive** is the string given to `omp(...)` ({doc}`decorator/directive`). This page
describes every directive that OMP4Py supports. The clauses that refine them are explained in
{doc}`clauses`, and directives that can be written together (`parallel for`, `parallel
sections`, ...) in {doc}`combined_directives`.

```{note}
Every example below is a complete function: copy it into a file and run it. Only supported
directives are listed here; see {doc}`/compatibility/coverage` for the full OpenMP 4.5
status.
```

## Summary

| Directive | Purpose | Needs a block? | Section |
| --- | --- | :-: | --- |
| `parallel` | Create a team of threads that all run the block | yes | [parallel](#parallel) |
| `for` | Split the iterations of a loop among the team | yes | [for](#for) |
| `sections` / `section` | Run independent blocks on different threads | yes | [sections](#sections-and-section) |
| `single` | One thread runs the block | yes | [single](#single) |
| `master` | Only thread 0 runs the block | yes | [master and masked](#master-and-masked) |
| `masked` | Only the selected thread runs the block | yes | [master and masked](#master-and-masked) |
| `critical` | One thread at a time runs the block | yes | [critical](#critical) |
| `barrier` | The whole team waits here | no | [barrier](#barrier) |
| `ordered` | Run part of a loop iteration in loop order | yes | [ordered](#ordered) |
| `threadprivate` | A global variable with one copy per thread | no | [threadprivate](#threadprivate) |
| `declare reduction` | Define your own reduction operator | no | [declare reduction](#declare-reduction) |

Directives that need a block are written `with omp("..."):`; the others are plain statements,
`omp("...")`.

## Rules for all directives

- Write them **inside code that is being processed**: a decorated function/class or a module of a
  package in package mode ({doc}`decorator/index`).
- The block of a directive is a *structured block*: you cannot `return`, `yield`, or
  `break`/`continue` out of it.
- Variables named in clauses must **exist before** the region.

## `parallel`

Creates a team of threads; every thread executes the block.

```{literalinclude} ../examples/directives/parallel.py
:language: python
```

```text
hello from thread 3 of 4
hello from thread 2 of 4
hello from thread 1 of 4
hello from thread 0 of 4
```

- The team size comes from `num_threads(n)`, `omp_set_num_threads` or `OMP_NUM_THREADS`
  ({doc}`runtime_api`, {doc}`environment`).
- `if(expr)` runs the region with a single thread when `expr` is false.
- The region ends with an implicit barrier: the threads join.
- Clauses: `if`, `num_threads`, `default`, `private`, `firstprivate`, `shared`, `reduction`,
  `copyin`, `proc_bind` ({doc}`clauses`).

## `for`

Distributes the iterations of the loop that follows among the threads of the current team. It
must be inside a `parallel` region (or combined with it, see {doc}`combined_directives`).

```{literalinclude} ../examples/directives/for_loop.py
:language: python
```

```text
[0, 1, 4, 9, 16, 25, 36, 49]
```

Rules for the loop:

- The block holds **only** the loop, and it must be `for i in range(...)` (1, 2 or 3 arguments).
- No `else` branch, and no `break`, `return` or `yield` that belong to that loop.
- With `collapse(n)` the `n` loops must be perfectly nested and the inner `range` bounds cannot
  use the outer loop variables.
- The loop variable is private automatically.
- Ends with an implicit barrier unless you add `nowait`.

Clauses: `private`, `firstprivate`, `lastprivate`, `reduction`, `schedule`, `collapse`,
`ordered`, `nowait`.

## `sections` and `section`

Each `section` block is run once, by one of the threads.

```{literalinclude} ../examples/directives/sections.py
:language: python
```

```text
{'a': 499500, 'b': 1999000}
```

Inside `sections` only `section` blocks are allowed. Clauses of `sections`: `private`,
`firstprivate`, `lastprivate`, `reduction`, `nowait`.

## `single`

The block is run by exactly one thread (any of them). The others wait at the implicit barrier
unless `nowait` is used. `copyprivate(x)` copies the value computed by that thread to the
private copies of all the others.

```{literalinclude} ../examples/directives/single.py
:language: python
```

```text
['computed once', 'computed once', 'computed once', 'computed once']
```

Clauses: `private`, `firstprivate`, `copyprivate`, `nowait`.

## `master` and `masked`

`master` runs the block only on thread 0, without any implied barrier. `masked` is the newer
form and accepts `filter(thread_id)` to select another thread.

```{literalinclude} ../examples/directives/master_masked.py
:language: python
```

```text
master runs on thread 0
masked runs on thread 2
```

(The order of the two lines may vary.)

## `critical`

Mutual exclusion: only one thread at a time runs the block. Unnamed `critical` blocks all share
one lock; `critical(name)` blocks with the same name share a lock, and blocks with different
names do not block each other.

```{literalinclude} ../examples/directives/critical.py
:language: python
```

```text
(4000, 8000)
```

## `barrier`

A stand-alone directive: every thread of the team waits until all have reached it.

```{literalinclude} ../examples/directives/barrier.py
:language: python
```

```text
['phase 1', 'phase 1', 'phase 1', 'phase 2', 'phase 2', 'phase 2']
```

All "phase 1" entries precede every "phase 2" entry: nobody passes the barrier early.

## `ordered`

Inside a loop that has the `ordered` clause, the block runs in the sequential order of the
iterations, while the rest of the iteration still runs in parallel.

```{literalinclude} ../examples/directives/ordered.py
:language: python
```

```text
[0, 1, 4, 9, 16, 25, 36, 49]
```

## `threadprivate`

Gives a **global** variable one persistent copy per thread. With the decorator, the directive is
written inside a decorated function; in {doc}`package mode <decorator/packages>` it can be at
module level.

```{literalinclude} ../examples/directives/threadprivate.py
:language: python
```

```text
thread 0 counter 100
thread 1 counter 101
thread 2 counter 102
master copy afterwards: 100
```

- Values persist between parallel regions (for the same threads).
- `copyin(x)` on `parallel` initialises each thread's copy from the master's value.
- A threadprivate variable cannot be listed in `private`, `shared`, ... clauses.

## `declare reduction`

Defines a **reduction operator of your own**, to use in `reduction(op: x)` clauses
({doc}`clauses`) for anything the built-in operators (`+ - * & | ^ and or`) do not cover: merging
lists, keeping the longest string, combining objects, `max`/`min` of custom data...

It is a stand-alone directive (no block). A declaration has three parts:

| Part | Meaning | Names you can use |
| --- | --- | --- |
| the operator name | The name you will write in `reduction(name: x)` | |
| the **combiner** | How a partial result is merged into the result | `omp_out` (the result) and `omp_in` (the partial value) |
| the **initializer** (optional) | The starting value of each thread's private copy | `omp_priv` (the private copy) and `omp_orig` (the original variable) |

### Two ways to write the combiner

The combiner can be written **inside** the parentheses, as in classic OpenMP, or **outside**, as a
clause:

```python
# combiner inside, after the type:     declare reduction(name : type : combiner)
omp("declare reduction(merge : list : omp_out += omp_in) initializer(omp_priv = [])")

# combiner outside, as a clause:       declare reduction(name : type) combiner(...)
omp("declare reduction(merge : list) combiner(omp_out += omp_in) initializer(omp_priv = [])")
```

Both are equivalent. The combiner is **a Python statement** written in terms of `omp_out` and
`omp_in`; the initializer is a statement that assigns `omp_priv`. The second form is clearer when the
combiner or the initializer are long: you can split the string over several lines.

```{literalinclude} ../examples/directives/declare_reduction.py
:language: python
```

```text
[0, 1, 2, 3, 4, 5, 6, 7]
abcd
```

How it works: every thread gets a private variable initialised by the initializer; the loop updates
that private copy; when the loop ends each private copy is merged into the original variable by
the combiner.

### Rules

- Declare the reduction **before** the directive that uses it, in the same function, or earlier in the
  same file (decorated functions of one file share their declarations; in
  {doc}`package mode <decorator/packages>` it can be written at module level).
- The **type** restricts the operator to variables with that annotation (`items: list = []`). Use
  `object` to make it available for any variable, annotated or not.
- Declaring the same name and type twice is an error: `redeclaration of 'merge' ...`.
- If you omit the initializer, each private copy starts as a **new, empty instance of the same type**
  as the variable (`type(x)()`: an empty list for a list, `0` for an int).

