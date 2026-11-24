# Debugging and Common Pitfalls

## Look at the generated code

The `dump` option writes the transformed source to a file. Give it a **path**, or `True` to use a
default name:

| Value | Where the code is written |
| --- | --- |
| `dump="pi_transformed.py"` | The path you give |
| `dump=True` (or `OMP4PY_DUMP=1`) | Next to the source file, named `<file>_<name>_omp_dump.py`, where `<name>` is the decorated function or class (for `f` in `prog.py`: `prog_f_omp_dump.py`). Whole modules processed in package mode use `<file>_omp_dump.py` |


```python
@omp(dump="pi_transformed.py", ignore_cache=True)
def pi(n):
    width = 1.0 / n
    total = 0.0
    with omp("parallel for reduction(+:total) firstprivate(width)"):
        for i in range(n):
            x = (i + 0.5) * width
            total += 4.0 / (1.0 + x * x)
    return total * width
```

The file `pi_transformed.py` contains (this is real output):

```python
def pi(n):
    width = 1.0 / n
    total = 0.0

    def _omp_parallel():
        nonlocal total, width
        _omp_1total: object
        _omp_1total = 0
        _omp_1width: object
        _omp_1width = _omp.copy_var(width)
        _omp_bounds: _omp.ForBounds = _omp.for_bounds(1)
        _omp_bounds.rs[0], _omp_bounds.rs[1], _omp_bounds.rs[2] = (0, n, 1)
        _omp.for_init(_omp_bounds, -1, 115, -1, 0)
        while _omp.for_next_static(_omp_bounds):
            for i in range(_omp_bounds.init, _omp_bounds.end):
                x = (i + 0.5) * _omp_1width
                _omp_1total += 4.0 / (1.0 + x * x)
        _omp.for_end(True)
        _omp.team_lock()
        total += _omp_1total
        _omp.team_unlock()
    _omp.parallel(_omp_parallel, True, (), 0, ())
    return total * width
```

How to read it:

- The body of the `parallel` region became the nested function `_omp_parallel`,
  executed by every thread through `_omp.parallel(...)`.
- `reduction(+:total)` created a private `_omp_1total` initialised to the identity;
  it is combined into `total` at the end.
- `firstprivate(width)` created `_omp_1width`, a private copy of `width` for each thread, and the loop reads
  that name.
- The `for` loop became a `while` over chunks handed out by the runtime
  (`for_next_static`); each chunk is iterated with a plain `range`.
- Names starting with `_omp` are generated. Do not use them in your own code.


With `debug=True` and `compile=True`, Cython's annotated HTML is kept next to the
build output, which helps find lines that still run through the Python C-API.

### Another example: `critical`

```python
@omp(dump="crit.py")
def crit():
    n = 0
    with omp("parallel"):
        with omp("critical"):
            n += 1
    return n
```

```python
def crit():
    n = 0

    def _omp_parallel():
        nonlocal n
        _omp.critical_lock()
        n += 1
        _omp.critical_unlock()
    _omp.parallel(_omp_parallel, True, (), 0, ())
    return n
```

The `parallel` region is always the same shape: a nested function plus one call to
`_omp.parallel(function, if_clause, num_threads, proc_bind, copyin)`. `critical` is a lock/unlock
pair around its block; named critical sections use a mutex created once at the top of the module.

## A debugging workflow

1. **Run with one thread** (`OMP_NUM_THREADS=1`, or `num_threads(1)`). If the result is already
   wrong, the problem is in the algorithm or the data sharing, not in concurrency.
2. **Use the pure runtime** (`OMP4PY_PURE=1`) so every frame is plain Python in the traceback and
   you can use a normal debugger.
3. **Dump the generated code** and read how your variables were treated (`_omp_1x` names are the
   private copies).
4. **Check the environment** with `OMP_DISPLAY_ENV=true`: team size, schedule, nesting.
5. **Print the thread number** (`omp_get_thread_num()`), protected by `critical`, to see who does
   what.
6. **Run several times.** A result that changes between runs usually means a data race.
7. **Disable the cache** (`ignore_cache=True`) when you change an extension or suspect stale code.

## Stale results? Clear the cache

Transformed functions are cached using a hash of the source file that defines them. If you change
an *extension* (a custom device, parser or modifier), the hash does not change and the
old transformation is reused. Pass `ignore_cache=True` (or set
`OMP4PY_IGNORE_CACHE=1`) while developing.

## Syntax errors

Directive problems are reported as ordinary `SyntaxError` exceptions that point at the
offending text inside your file:

| Message | Cause |
| --- | --- |
| `omp () argument needs constant string expression` | The argument is not a plain string literal. |
| `omp () can only be used as a statement or in a 'with' block` | The call is inside an expression. |
| `cannot use more than one directive item in a 'with' statement` | `with omp("..."), other():` |
| `'x' not implemented` | The construct is parsed but has no transformation yet. |
| `invalid branch to/from OpenMP structured block` | `return`, `yield` or an escaping `break`/`continue` in the block. |
| `name 'x' is not defined` | A data-sharing clause names a variable that does not exist yet. |
| `'x' appears more than once in data clauses` | The same variable twice in `private`/`shared`/`reduction`. |
| `'x' is predetermined 'threadprivate'` | A threadprivate variable was used in a data-sharing clause. |
| `for statement expected` | The block of `for` is not a single `for` loop. |
| `invalid controlling predicate` | The loop is not `for ... in range(...)`. |
| `expected 'omp section'` | Something other than `section` blocks inside `sections`. |
| `Invalid decorator order: expected the 'omp' decorator to be the innermost.` | `@omp` is not the last decorator. |
| `loop cannot have an else block` | The `for` of a worksharing loop has an `else`. |
| `'break' statement used with OpenMP for loop` | `break`, `return` or `yield` that belongs to the loop. |
| `collapse argument needs positive constant integer expression` | `collapse(n)` with `n` not a positive integer literal. |
| `not enough perfectly nested loops before statement` | `collapse(n)` with fewer than `n` nested loops. |
| `collapsed loops not perfectly nested` | Statements between the loops of a `collapse`. |
| `expression refers to iteration variable` | A collapsed inner `range` uses an outer loop variable. |
| `schedule 'auto' does not take a 'chunk_size' parameter` | `schedule(auto, 4)`. |
| `A structured block is required; the directive must be defined using a 'with' block.` | A directive that needs a block was written as a plain statement. |
| `'x' must be 'threadprivate' for 'copyin'` | `copyin` of a variable not declared `threadprivate`. |
| `user defined reduction not found for 'x'` | `reduction(op:x)` with an operator that is not built in and was not declared with `declare reduction` (or was declared after its use). |
| `Unknown transformation device 'name'` | `device="name"` that is not registered ({doc}`/contributor/extensions/devices`). |
| `Unknown directive parser 'name'` | A name in `parsers` that is not registered ({doc}`/contributor/extensions/subparsers`). |

## Behavioural pitfalls

**Output from several threads interleaves.** Wrap the print in `critical`, or collect
the results and print after the region.

**Shared writes without protection.** `counter += 1` from several threads is a race.
Use `reduction(+:counter)` or `critical`.

**`private` does not copy the value.** Use `firstprivate` when the threads need the
starting value.

**Variables created inside the region.** They are local to the generated function.
Define them before the region, or return them with `lastprivate`/`reduction`.

**Exceptions inside a region.** An exception raised in a thread propagates out of the
parallel region (the test-suite checks this with `ArithmeticError`).

**Directives in `exec()` or the REPL.** The preprocessor needs source code. Functions
defined from `exec()` strings or interactively fail with `OSError: could not get
source code`; put the code in a file.

**`range` only.** `for x in my_list` cannot be used under a worksharing `for`: iterate over indices
(`for i in range(len(my_list))`) and use `my_list[i]`.

**Results differ in the last digits.** Floating-point addition is not associative: the order in
which partial sums are combined by a `reduction` depends on the threads. Compare with a
tolerance, not with `==`.

**`nowait` and later reads.** After `for ... nowait` the other threads may still be working.
Add `omp("barrier")` before reading results produced by other threads.

**A variable is not what I expect inside the region.** Remember the default: variables defined
outside are shared; loop variables are private; names in `private` start uninitialised.
`default(none)` lists every use and finds forgotten ones.
