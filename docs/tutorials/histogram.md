# Tutorial: Word Histogram with `critical`

Counting words is not a numerical reduction: the shared result is a dictionary-like
object. This makes it a good place to learn about mutual exclusion.

```{literalinclude} ../examples/histogram.py
:language: python
```

```text
[('alpha', 1000), ('beta', 1500), ('gamma', 500)]
[('alpha', 1000), ('beta', 1500), ('gamma', 500)]
```

## Version 1: `critical` around every update

```python
with omp("parallel for schedule(dynamic, 64) num_threads(4)"):
    for i in range(len(words)):
        with omp("critical"):
            counts[words[i]] += 1
```

`counts[word] += 1` reads and writes the same container, so two threads updating at once
could lose an increment. `critical` guarantees that only one thread at a time runs the
update. It is correct, but every iteration waits for the lock: the program is
essentially serial.

## Version 2: private partial results

```python
with omp("parallel num_threads(4)"):
    local = Counter()
    with omp("for schedule(static) nowait"):
        for i in range(len(words)):
            local[words[i]] += 1
    with omp("critical"):
        counts.update(local)
```

Here:

1. `local` is created **inside** the `parallel` region, so each thread owns one.
2. The `for` loop updates only the private `local`; no lock is needed.
3. `nowait` removes the barrier at the end of the loop: a thread can go straight to the
   merge.
4. `critical` runs once per thread (4 times in total) instead of once per word.

```{mermaid}
sequenceDiagram
    participant T0 as Thread 0
    participant T1 as Thread 1
    participant C as counts (shared)
    par count locally
        T0->>T0: local₀ updated N/4 times
    and
        T1->>T1: local₁ updated N/4 times
    end
    T0->>C: critical: counts.update(local₀)
    T1->>C: critical: counts.update(local₁)
```

This pattern, *privatise, compute, merge once*, is the manual version of what
`reduction` does automatically for numbers.

## Named critical sections

Unnamed `critical` blocks all share one lock. Use names when unrelated resources must not
block each other:

```python
with omp("critical(stats)"):
    stats.update(...)
with omp("critical(log)"):
    log.append(...)
```
