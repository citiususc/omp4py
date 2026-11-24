# Tutorial: Longest Collatz Chain

Start from any positive integer `n`. If it is even, divide it by two; if it is odd, multiply it by
three and add one. Repeat until you reach 1. The number of steps this takes is the length of the
*Collatz chain* of the starting number. For example, `6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1` has
8 steps.

The question: **which starting number below a limit has the longest chain?**

## Why this needs a custom reduction

Computing every chain is independent work, so `parallel for` fits. But the *result* is not a sum: we
want **the largest length and the number that produced it**, a pair `(steps, start)`. The built-in
reduction operators (`+ - * & | ^ and or max min`) cannot express that: `max` and `min` keep the value, not where it was found. With `declare reduction` we define the
operator ourselves ({doc}`/user_guide/directives`).

## Sequential version

```python
def longest_collatz(limit):
    best = (0, 0)                      # (steps, starting number)
    for start in range(1, limit):
        n = start
        steps = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        if steps > best[0]:
            best = (steps, start)
    return best
```

## Parallel version

```{literalinclude} ../examples/collatz.py
:language: python
```

```text
below 100000, the longest chain starts at 77031 and has 350 steps
```

## The three pieces

**The operator.** The `declare reduction` directive has the name (`best`), the type it applies to
(`tuple`), a **combiner** and an **initializer**:

```python
omp("declare reduction(best : tuple) "
    "combiner(omp_out = omp_in if omp_in[0] > omp_out[0] else omp_out) "
    "initializer(omp_priv = (0, 0))")
```

- `omp_out` is the result so far and `omp_in` is a partial result of another thread: the combiner keeps
  the pair with more steps.
- `omp_priv` is the starting value of every thread's private copy: `(0, 0)`, a pair that any real chain
  beats.

**The variable.** `best: tuple = (0, 0)` has the **annotation** `tuple`. That is what makes the
reduction declared for `tuple` apply to this variable ({doc}`/user_guide/directives`).

**The clause.** `reduction(best:best)` reads "reduce the variable `best` with the operator `best`". Each
thread updates its private copy, and when the loop ends the private copies are merged into the
original with the combiner, one thread at a time.

## Why `schedule(dynamic, 500)`

Chains have very different lengths: a number can reach 1 in 10 steps and the next one in 300. Giving the
threads blocks of 500 numbers on demand keeps them all busy, and it is cheap because a block is a lot
of work.

```{mermaid}
flowchart TB
    L["numbers 1 to limit"] --> B1["block of 500"]
    L --> B2["block of 500"]
    L --> B3["more blocks"]
    B1 --> T0["thread 0: best 0"]
    B2 --> T1["thread 1: best 1"]
    B3 --> T2["thread k: best k"]
    T0 --> M["combiner: keep the longest"]
    T1 --> M
    T2 --> M
    M --> R["best = (350, 77031)"]
```

## Notes

- The loop body only uses local variables and integers, which makes this a good example for
  free-threaded Python: the threads share nothing while they work.
- Below 100000 the longest chain is unique, so the result is the same on every run. When several
  numbers tie for the longest chain, the combiner keeps the pair that is already in the result, which
  depends on the order in which the private copies are merged: see exercise 1.

## Exercises

1. Make the combiner break ties towards the smallest starting number.
2. Keep three values at the same time, with a triple `(total, steps, start)` that adds up all the chain
   lengths and keeps the maximum.
3. Remove the annotation `: tuple` from `best`. You get `user defined reduction not found for 'best'`:
   the operator was declared for `tuple`, and an unannotated variable does not match it. Declare it for
   `object` to make it work for any variable.
