# Tutorial: Matrix Multiplication

Matrix multiplication is a good example of *embarrassingly parallel* work: each element
`c[i][j]` can be computed independently.

```{literalinclude} ../examples/matmul.py
:language: python
```

```text
max error: 1.4210854715202004e-14
```

(The exact error depends on your platform; it should be around `1e-14`.)

## Why `collapse(2)`

With 64 rows and 4 threads, parallelising only the outer loop gives each thread 16
rows, which balances well. With a *tall and thin* result (say 3 rows and thousands of
columns) the outer loop would offer only 3 units of work. `collapse(2)` fuses the two
loops into one iteration space of `rows × cols` items before dividing it among the
threads:

```{mermaid}
flowchart LR
    A["for i in range(n)<br/>for j in range(n)"] -->|collapse 2| B["one space of n×n<br/>iterations"]
    B --> C["thread 0: first chunk"]
    B --> D["thread 1: second chunk"]
    B --> E["thread 2: third chunk"]
    B --> F["thread 3: fourth chunk"]
```

The rules for `collapse` (checked at preprocessing time):

- The loops must be perfectly nested (nothing between the two `for` lines).
- Inner loop bounds cannot depend on the outer loop variable.
- Each loop must be a `range(...)` loop.

## Why no clauses for the data

| Variable | Sharing | Reason |
| --- | --- | --- |
| `a`, `b` | shared (default) | Only read. |
| `c` | shared (default) | Each thread writes different elements: no conflict. |
| `i`, `j` | private (automatic) | Loop variables of the worksharing loops. |
| `acc`, `k` | per iteration | Assigned and used inside the loop body (local to the thread). |

```{tip}
`acc` is created inside the loop body, so each iteration works with a fresh local.
If you needed `acc` after the loop, you would declare it before the region and add
`lastprivate(acc)` or a `reduction`.
```

## Speeding it up

Lists of lists are flexible but slow. Two typical improvements:

- Compile with types and use flat lists or `array`/NumPy buffers.
- Transpose `b` once so the innermost loop reads both operands sequentially.
