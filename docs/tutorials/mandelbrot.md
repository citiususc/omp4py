# Tutorial: The Mandelbrot Set

A point `c` of the complex plane belongs to the Mandelbrot set if the sequence
$z_{k+1} = z_k^2 + c$, starting at $z_0 = 0$, never leaves the disc $|z| \le 2$. We cannot iterate
forever, so for every point we count how many iterations it takes to escape, up to a maximum. That
*escape time* gives the colour of the pixel.

This is a classic parallel example because **every pixel is independent**, but the amount of work
per pixel is **very uneven**.

## Sequential version

```python
def escape_time(c, max_iter):
    z = 0j
    n = 0
    while n < max_iter and abs(z) <= 2.0:
        z = z * z + c
        n += 1
    return n


def mandelbrot_sequential(width, height, max_iter=100):
    return [
        [escape_time(complex(-2.0 + 3.0 * x / width, -1.0 + 2.0 * y / height), max_iter) for x in range(width)]
        for y in range(height)
    ]
```

A point far from the set escapes after one or two iterations. A point inside the set never escapes and
needs all `max_iter` iterations. The rows that cross the set are therefore much more expensive than
the rows that do not.

## Parallel version

```{literalinclude} ../examples/mandelbrot.py
:language: python
```

The program checks the parallel image against the sequential one and prints it as text:

```text
True
                                        . .                     
                                       ...                      
                                     ..@@-..                    
                                     :@@@@@.                    
                              .:. .:...@@@.....   .             
                              .-@:@@@@@@@@@@@@@.-+:.            
                             ..@@@@@@@@@@@@@@@@@@@.             
                            .@@@@@@@@@@@@@@@@@@@@@%             
                ... ..     .+@@@@@@@@@@@@@@@@@@@@@@@-           
                 .@@@@@@:..+@@@@@@@@@@@@@@@@@@@@@@@..           
               .-+@@@@@@@@.@@@@@@@@@@@@@@@@@@@@@@@@.            
             ..:-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@             
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.              
             ..:-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@             
               .-+@@@@@@@@.@@@@@@@@@@@@@@@@@@@@@@@@.            
                 .@@@@@@:..+@@@@@@@@@@@@@@@@@@@@@@@..           
                ... ..     .+@@@@@@@@@@@@@@@@@@@@@@@-           
                            .@@@@@@@@@@@@@@@@@@@@@%             
                             ..@@@@@@@@@@@@@@@@@@@.             
                              .-@:@@@@@@@@@@@@@.-+:.            
                              .:. .:...@@@.....   .             
                                     :@@@@@.                    
                                     ..@@-..                    
                                       ...                      
```

What changed with respect to the sequential code:

1. `@omp` on the function.
2. `with omp("parallel for ...")` around the outer loop: each thread computes whole rows.
3. `schedule(dynamic)`, explained below.

Nothing else: `escape_time` is an ordinary function that is **not** decorated, and the parallel
region can call it, use `complex` numbers, `abs`, and write into a list of lists.

## Why `schedule(dynamic)`

With the default `static` schedule the rows are split into equal blocks, one per thread. The threads
that get the middle rows (inside the set) have far more work than the others, and the rest wait
idle at the end of the loop. With `dynamic`, a thread asks for the next row when it finishes the
previous one, so the work balances itself:

```{mermaid}
flowchart LR
    subgraph static["static"]
        direction TB
        A["thread 0<br/>rows 0-5, cheap"]
        B["thread 1<br/>rows 6-11, expensive"]
        C["thread 2<br/>rows 12-17, expensive"]
        D["thread 3<br/>rows 18-23, cheap"]
    end
    subgraph dynamic["dynamic"]
        direction TB
        Q["next free row"] --> E["thread 0"]
        Q --> F["thread 1"]
        Q --> G["thread 2"]
        Q --> H["thread 3"]
    end
```

Use `schedule(dynamic, chunk)` with a larger `chunk` when each row is cheap, to reduce the cost of
asking for work.

## Why there is no data race

Every iteration of the outer loop writes only to its own row, `image[y]`, so two threads never write
the same element. The list `image` is shared, and that is what we want: no `reduction` or `critical`
is needed.

## Exercises

1. Replace `schedule(dynamic)` by `schedule(static)` and measure the time of each version on a large
   image (for example `mandelbrot(1200, 800, 200)`).
2. Add `collapse(2)`: the two loops are perfectly nested, so the `width * height` pixels are shared
   among the threads instead of the rows. Does the balance improve?
3. Change `num_threads(4)` to `omp_get_max_threads()` and see how the time evolves with the
   number of threads. This depends on your interpreter, see {doc}`/user_guide/performance`.
