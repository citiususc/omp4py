# Clauses

Clauses refine a directive. They are written after the construct, separated by spaces, and many
accept **modifiers**:

```python
with omp("parallel for num_threads(4) schedule(monotonic: dynamic, 8) reduction(+:total) private(tmp)"):
```

The directive sections of this guide ({doc}`directives`) list which clauses each directive
accepts. This page explains what each clause **does**, grouped by purpose.

```{contents}
:local:
:depth: 2
```

## Cheat sheet

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - Clause
  - What it does
  - Example
* - `shared(x, y)`
  - One variable seen by all threads
  - `shared(table)`
* - `private(x)`
  - Each thread has its own, **uninitialised** copy
  - `private(tmp)`
* - `firstprivate(x)`
  - Private copy **initialised** from the original
  - `firstprivate(offset)`
* - `lastprivate(x)`
  - Private; the last iteration/section copies its value back
  - `lastprivate(last)`
* - `default(shared\|none\|private\|firstprivate)`
  - Sharing of variables not listed elsewhere
  - `default(none)`
* - `reduction(op: x)`
  - Private partials combined with `op` at the end
  - `reduction(+:total)`
* - `copyin(x)`
  - Initialise `threadprivate` copies from the master
  - `copyin(counter)`
* - `copyprivate(x)`
  - Broadcast a `single` result to all threads
  - `copyprivate(token)`
* - `if(cond)`
  - Run in parallel only when `cond` is true
  - `if(n > 1000)`
* - `num_threads(n)`
  - Size of the team
  - `num_threads(8)`
* - `schedule(kind[, chunk])`
  - How loop iterations are handed out
  - `schedule(dynamic, 16)`
* - `collapse(n)`
  - Fuse `n` nested loops into one
  - `collapse(2)`
* - `ordered`
  - Allow `ordered` blocks in the loop
  - `parallel for ordered`
* - `nowait`
  - Remove the implicit barrier at the end
  - `for nowait`
* - `filter(id)`
  - Thread that runs a `masked` block
  - `masked filter(2)`
```

## Data-sharing clauses

They decide whether the threads share a variable or each has its own. **The variables must exist
before the region.**

```{literalinclude} ../examples/clauses/data.py
:language: python
```

```text
(4, 10, 5, 9)
```

Read the result as: `shared_value` was incremented once per thread (4); `private_value` kept its
original `10` because each thread changed only its own copy; `first` kept `5` (each thread
incremented its private copy, which started at 5); and `last` received `9`, the value of the last
iteration of the loop.

| Clause | Copy per thread? | Initial value | Value after the region |
| --- | :-: | --- | --- |
| `shared` | no | the variable's | updated by whoever wrote |
| `private` | yes | **uninitialised** | unchanged |
| `firstprivate` | yes | the original's | unchanged |
| `lastprivate` | yes | uninitialised | value of the last iteration/section |

`default(none)` forces you to list every variable used in the region, which helps to catch
mistakes; `default(private)` and `default(firstprivate)` make unlisted variables private.

### Reductions

`reduction(op:x)` gives each thread a private copy initialised to the **identity** of the
operator, and combines the copies into the original variable when the construct ends:

```{literalinclude} ../examples/clauses/reduction.py
:language: python
```

```text
(15, 32, False)
```

Built-in operators:

| Operator | Identity | Combines with |
| --- | --- | --- |
| `+` | `0` | `+=` |
| `-` | `0` | `-=` |
| `*` | `1` | `*=` |
| `&` | all bits set | `&=` |
| `\|` | `0` | `\|=` |
| `^` | `0` | `^=` |
| `and` | `True` | `and` |
| `or` | `False` | `or` |

Several reductions can be given in one directive as separate clauses
(`reduction(+:a) reduction(*:b)`). A variable cannot appear in two reductions.

**Your own operators.** For anything else (merging lists, `max`/`min`, custom objects) declare
the operator with the `declare reduction` directive ({doc}`directives`) and use its name:
`reduction(merge: items)`.

## Control clauses

```{literalinclude} ../examples/clauses/control.py
:language: python
```

```text
n = 10 -> team of 1
n = 1000 -> team of 4
```

- **`if(cond)`**: when `cond` is false the region is executed by a single thread. Use it to
  avoid the cost of a team for small problems.
- **`num_threads(n)`**: team size. It accepts an expression evaluated when the region starts.
  A tuple (`num_threads((4, 2))`) gives one size per nesting level.
- **`nowait`**: removes the barrier at the end of `for`, `sections` or `single`, so threads that
  finish early continue immediately. Add an explicit `omp("barrier")` when you need to wait later.
- **`filter(id)`**: selects the thread that runs a `masked` block.
- **`proc_bind(...)`**: accepted, but thread affinity is not implemented yet, so it has no
  effect.

## Loop scheduling

`schedule([modifier:] kind[, chunk])` decides how the iterations of a `for` are assigned to
threads:

| Kind | Behaviour | Good for |
| --- | --- | --- |
| `static` | Chunks are assigned in advance, round-robin. Without a chunk size, one contiguous block per thread. | Uniform iterations; lowest overhead |
| `dynamic` | A thread takes the next chunk when it finishes one. The default chunk is 1. | Irregular iterations |
| `guided` | Like dynamic, with chunks that shrink. The default chunk is 1. | Smoothly decreasing work |
| `auto` | Left to the implementation (it uses `static`). No chunk allowed. | |
| `runtime` | Taken from `OMP_SCHEDULE` or `omp_set_schedule` | Experiments without recompiling |

```{literalinclude} ../examples/clauses/schedule.py
:language: python
```

```text
static,2 : [0, 0, 1, 1, 2, 2, 0, 0]
runtime  : [0, 0, 0, 0, 1, 1, 1, 1]
```

Each position shows which thread ran the iteration. With `static, 2` the chunks of two
iterations rotate among the three threads.

Modifiers of `schedule`: `monotonic` (each thread receives increasing iterations) and
`nonmonotonic`. For example, `schedule(monotonic: dynamic, 2)`.

### `collapse` and `ordered`

```{literalinclude} ../examples/clauses/collapse_nowait.py
:language: python
```

```text
20
```

`collapse(2)` fuses the two loops into a single iteration space of `n * m` items before the
schedule is applied, which gives more work to distribute. `ordered` is shown with the
`ordered` directive in {doc}`directives`.

## Clause modifiers

Some clauses accept a **modifier** before a colon:

| Modifier | Where | Meaning |
| --- | --- | --- |
| `monotonic`, `nonmonotonic` | `schedule(...)` | Iteration order guarantee |
| the reduction operator | `reduction(+: x)` | The operator itself (`+`, `*`, `and`, ...) |
| a construct name | most clauses of a combined directive | Which construct of the directive receives the clause, e.g. `if(parallel: cond)` |

The last one is described in {doc}`combined_directives`.
