# Tutorial: Computing Pi

We approximate

$$\pi = \int_0^1 \frac{4}{1 + x^2}\,dx$$

with the midpoint rule: split `[0, 1]` into `n` slices, evaluate the function at the
centre of each slice and add the areas.

## Sequential version

```python
def pi(n):
    width = 1.0 / n
    total = 0.0
    for i in range(n):
        x = (i + 0.5) * width
        total += 4.0 / (1.0 + x * x)
    return total * width
```

Every iteration is independent except for the accumulation into `total`.

## Parallel version

```{literalinclude} ../examples/pi.py
:language: python
```

```text
3.1415926535898753
```

Four changes were needed:

1. `@omp` on the function.
2. `with omp("parallel for ...")` around the loop.
3. `reduction(+:total)`, so every thread accumulates privately.
4. `firstprivate(width)`, so every thread reads its own local name for `width` in every iteration
   ({doc}`/user_guide/performance` explains why this matters).

Without the `reduction` clause, `total += ...` would be a data race and the result would
be wrong (and different on every run).

## What the transformer produces

The reduction clause creates a private copy per thread and merges it at the end
(see {doc}`/user_guide/debugging` for the full generated code):

```{mermaid}
flowchart TB
    S["total = 0.0 (shared)"] --> F["fork: 4 threads"]
    F --> T0["thread 0: partial₀ = Σ over its chunk"]
    F --> T1["thread 1: partial₁"]
    F --> T2["thread 2: partial₂"]
    F --> T3["thread 3: partial₃"]
    T0 --> J["join: total += partial_k"]
    T1 --> J
    T2 --> J
    T3 --> J
    J --> R["return total * width"]
```

## Checking correctness

Compare with `math.pi`, and with the sequential answer using a small tolerance, because
adding the partial sums in a different order changes the last digits:

```python
import math
print(abs(pi(1_000_000) - math.pi) < 1e-9)    # True
```

## Exercises

1. Remove `reduction(+:total)` and run several times. What changes?
2. Replace it with `with omp("critical"): total += ...` inside the loop. Is it correct? Is it fast?
3. Add `schedule(dynamic, 1000)` and compare timings with `schedule(static)`.
4. Run the compiled-with-types variant ({doc}`/user_guide/execution_modes`) and compare.
