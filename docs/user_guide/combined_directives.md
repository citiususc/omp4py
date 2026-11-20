# Combined Directives

```{admonition} OMP4Py parses OpenMP 6.0 syntax
:class: tip

OMP4Py's directive parser is not limited to `parallel for`. It understands the **combined and
composite directive names of the OpenMP 6.0 specification** (the full list of *Conforming
Compound Directive Names*) and the **directive-name modifiers** of the clauses, so you can
write the same long directives you would write in C, C++ or Fortran. Which constructs inside
them can be *run* grows with each release ({doc}`/compatibility/coverage`); the syntax is
already there.
```

A **combined directive** puts several constructs in one directive. They are written one after
the other, from the **outermost to the innermost**, and the clauses of the whole directive are
distributed among them:

```python
with omp("target teams distribute parallel for simd if(target: use_gpu) num_threads(8) reduction(+: s)"):
    for i in range(n):
        ...
```

That single line stands for six nested constructs (`target`, `teams`, `distribute`, `parallel`,
`for`, `simd`) and eight clauses.

```{contents}
:local:
:depth: 2
```

## `masked`, not `master`

OpenMP 6.0 **deprecates `master`** in favour of `masked`. OMP4Py follows the 6.0 syntax, so
**combined directives must use `masked`**; `master` can only be used as a stand-alone
directive:

```python
with omp("parallel masked num_threads(4)"):     # valid
    ...

with omp("parallel master num_threads(4)"):     # rejected: expected clause, found 'master'
    ...
```

`masked` is a drop-in replacement (`masked` alone runs on thread 0, and `filter(id)` chooses another
thread), see {doc}`directives`.

## What you can write

Any directive name from the OpenMP 6.0 appendix is valid syntax. Some of them, grouped by purpose:

| Family | Examples |
| --- | --- |
| Parallel worksharing | `parallel for`, `parallel sections`, `parallel single`, `parallel masked` |
| With SIMD | `for simd`, `parallel for simd`, `distribute simd`, `distribute parallel for simd` |
| Teams and distribution | `teams distribute`, `teams distribute parallel for`, `distribute parallel for` |
| Offloading | `target parallel for`, `target teams distribute`, `target teams distribute parallel for`, `target teams distribute parallel for simd` |
| Tasks | `parallel taskloop`, `parallel masked taskloop`, `teams taskloop` |

Invalid combinations are rejected with a precise `SyntaxError` (for example, an unknown
construct in the middle, or a clause that none of the constructs accepts).

### Which ones run today

Parsing is complete; **running** needs a transformation for every construct of the directive.
Today OMP4Py can run the combinations built from supported constructs:

| Directive | Runs today? |
| --- | :-: |
| `parallel for`, `parallel sections`, `parallel single`, `parallel masked` | ✅ |
| `for simd`, `parallel for simd` | not yet (needs `simd`) |
| `parallel taskloop`, `parallel masked taskloop` | not yet (needs `taskloop`) |
| `teams distribute parallel for`, `distribute parallel for` | not yet (needs `teams`, `distribute`) |
| `target parallel for`, `target teams distribute parallel for [simd]` | not yet (needs `target`) |

A combination with a construct that has no transformation yet stops at preprocessing with a clear
message; the directive itself was understood:

```{literalinclude} ../examples/directives/combined_unsupported.py
:language: python
```

```text
SyntaxError: 'target' not implemented
```

As constructs are added ({doc}`/compatibility/coverage`), the combinations that contain them
start working **without any change to your directive strings**.

## Clauses are distributed among the constructs

Each clause is given to the construct(s) that accept it. This is how the directive above is
split. You write one line; OMP4Py builds the constructs and hands every clause to its owner:

```python
with omp("target teams distribute parallel for "
         "if(target: use_gpu) if(parallel: n > 1000) "
         "num_teams(4) thread_limit(64) num_threads(8) "
         "dist_schedule(static, 128) schedule(dynamic) collapse(2) "
         "reduction(+: s) map(tofrom: a) private(t) nowait"):
    ...
```

```{mermaid}
flowchart LR
    subgraph D["one directive string"]
        direction TB
        C1["if(target: ...)<br/>map(...)<br/>nowait"]
        C2["num_teams<br/>thread_limit<br/>reduction"]
        C3["dist_schedule<br/>collapse"]
        C4["if(parallel: ...)<br/>num_threads"]
        C5["private<br/>reduction<br/>schedule<br/>collapse"]
    end
    C1 --> T["target"]
    C2 --> TE["teams"]
    C3 --> DI["distribute"]
    C4 --> P["parallel"]
    C5 --> F["for"]
    T --> TE --> DI --> P --> F
```

The rules (these are the result of parsing the directive above, not a simplification):

| Clause | Goes to | Rule |
| --- | --- | --- |
| `map` | `target` | Only `target` accepts it |
| `num_teams`, `thread_limit` | `teams` | Only `teams` accepts them |
| `dist_schedule` | `distribute` | Only `distribute` accepts it |
| `num_threads` | `parallel` | Only `parallel` accepts it |
| `schedule` | `for` | Only `for` accepts it |
| `private` | the **innermost** construct that accepts it (`for`) | Innermost rule |
| `nowait` | the **outermost** construct that accepts it (`target`) | Outermost rule |
| `collapse` | `distribute` **and** `for` | Accepted by both |
| `reduction` | `teams` **and** `for` | Accepted by both |
| `if` | `target` **and** `parallel` | Accepted by both, unless a modifier chooses (below) |

When a clause is accepted by several constructs and nothing says otherwise, **every construct that
accepts it receives it**, as OpenMP specifies.

## Directive-name modifiers: choose which construct gets the clause

Look at the last row: `if` is accepted by **two** constructs of the directive, `target` and
`parallel`. A clause can start with a **directive-name modifier**, the name of one construct
followed by a colon, to say which one it is for. Compare these directives:

```python
with omp("target parallel for if(use_gpu)"):               # no modifier
with omp("target parallel for if(target: use_gpu)"):       # only target
with omp("target parallel for if(parallel: n > 1000)"):    # only parallel
```

| Directive | `target` receives `if` | `parallel` receives `if` |
| --- | :-: | :-: |
| `target parallel for if(use_gpu)` | ✔ | ✔ |
| `target parallel for if(target: use_gpu)` | ✔ | |
| `target parallel for if(parallel: n > 1000)` | | ✔ |

With the modifier only the named construct is affected. Without it, both are.

This gives you independent control over each level in the same directive, for example "offload
only when the device is available, and use several host threads only when the loop is long":

```python
with omp("target parallel for if(target: use_gpu) if(parallel: n > 1000) num_threads(8)"):
```

The same modifier works for any clause that several constructs accept (`num_threads(parallel: 8)`).
Naming a construct that does not accept the clause, or one that is not part of the directive, is
an error:

```text
target parallel for if(for: x)      ->  SyntaxError: 'if' is not valid for 'for'
parallel for if(simd: x)            ->  SyntaxError: 'simd' is not valid for 'parallel for'
```

Other modifiers exist for specific clauses, for example the ordering of `schedule`:
`schedule(monotonic: dynamic, 2)` ({doc}`clauses`).

## Running examples

What runs today, with the directive-name modifier included:

```{literalinclude} ../examples/directives/combined.py
:language: python
```

```text
(45, ['masked: only thread 0', 'single: exactly one thread'])
```

Here `if(parallel: n > 100)` is applied to the `parallel` construct: the region runs with one thread
when `n` is small and with three when it is large.

## Combined or nested?

- Use a **combined** directive when the constructs are adjacent: it is shorter and lets the
  clauses be written once.
- Use **nested** directives when you need code **between** the constructs (private setup after
  `parallel` and before `for`, a `critical` inside, ...):

```python
with omp("parallel num_threads(4)"):
    setup_per_thread()                 # runs once in every thread
    with omp("for schedule(dynamic)"):
        for i in range(n):
            ...
```
