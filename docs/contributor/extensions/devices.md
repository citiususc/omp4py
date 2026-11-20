# Devices

The default transformer is the **`omp` device**: the set of functions that turn each construct
(`parallel`, `for`, `critical`, ...) into Python code that calls the OMP4Py runtime. A **device**
is a function with the same job. Register your own when you want **your own transformations**:
for some constructs, or for all of them.

## The contract

```python
Device = Callable[[Construct, StructuredBlock | None, Context], list[ast.stmt]]
```

| Parameter | Meaning |
| --- | --- |
| `ctr` | One construct with its clauses as attributes (`Parallel`, `For`, `Critical`, or one of yours). |
| `block` | The user's structured block, or `None` for a stand-alone directive. |
| `ctx` | The transformation {py:class}`~omp4py.core.preprocessor.transformers.context.Context`. |
| *returns* | The statements that replace the directive. **They must include `block`** when one was given. |

The device is called **once per construct, from the outermost to the innermost**: for `parallel
for` first with `Parallel`, then, when the returned statements are visited again, with `For`.

```python
from omp4py.core.ext import add_device

@add_device("mydev")
def mydev(ctr, block, ctx):
    ...
```

Select it with `@omp(device="mydev")`, or for a package `omp(pkg=__package__, device="mydev")`.

```{note}
The name `"omp"` is reserved for the built-in device and cannot be replaced. Requesting an
unknown device raises `SyntaxError: Unknown transformation device ... Available devices: [...]`.
```

## Base case: delegate to `omp`

A device receives **all** the constructs, not only the ones it cares about. The usual pattern is
to write your own transformation for a few and **delegate the rest to `omp`**, through
`omp_construct`:

```{literalinclude} ../../examples/ext/loud_device.py
:language: python
```

```{literalinclude} ../../examples/ext/loud_user.py
:language: python
```

```text
only one thread prints this
--- barrier reached ---
--- barrier reached ---
```

Only `barrier` is transformed by our device (it adds a message and then still uses the standard
barrier); `parallel`, `single` and everything else are delegated, so the program behaves as usual.

## Overriding: your transformation replaces the default

If you **do not** call `omp_construct`, your transformation *replaces* the standard one. A device
that runs a program sequentially:

```{literalinclude} ../../examples/ext/serial_device.py
:language: python
```

```{literalinclude} ../../examples/ext/serial_user.py
:language: python
```

```text
4950
```

- For `Parallel` and `For` it returns `[block]`: the directive disappears and the body stays.
- `Barrier` has no block, so `[]` removes it.
- Everything else is delegated.

```{important}
If you drop a construct and keep the user's code, return `[block]`. Returning `[]` for a
construct that had a block **discards the user's code**.
```

## A dispatch table, like `omp`

When a device handles many constructs it is cleaner to mirror how `omp` works, with a
`singledispatch` function:

```{literalinclude} ../../examples/ext/dispatch_device.py
:language: python
```

```{literalinclude} ../../examples/ext/dispatch_user.py
:language: python
```

```text
[0, 0, 0, 1, 1, 1]
```

The user asked for `schedule(dynamic, 1)`; the device edits the construct (it is a dataclass:
`ctr.schedule = None`) and delegates, so the schedule becomes the default `static` and the run is
reproducible. **Editing the construct and then delegating** is the cheapest way to build
variations of the standard behaviour.

## Writing a transformation from scratch

Helpers used by the built-in transformers:

| Helper | Where | Use |
| --- | --- | --- |
| `ctr.span.to_ast(node)` | `Span` | Copy the directive's location to a generated node |
| `ast.fix_missing_locations(node)` | `ast` | Fill positions of children |
| `syntax_error_ctx(msg, span, ctx)` | `omp4py.core.ext` | A located `SyntaxError` |
| `with_block(ctr, block, ctx)` | `...transformers.transformer` | Require a `with` block and check control flow |
| `omp_name(ctx, "hint")` | `...transformers.symtable` | A unique generated identifier |
| `runtime_ast("name")` | `...transformers.symtable` | The AST of `_omp.name`, to call the OMP4Py runtime |
| `create_scope(...)`, `check_scopes(...)` | `...transformers.scopes` | Data-sharing semantics for a block turned into a function |

```{note}
Only `Modifier`, `add_*`, `omp_construct` and `syntax_error_ctx` are exported by
`omp4py.core.ext`. The helpers of the other modules are importable but internal and may change.
```

## Devices and custom constructs

A device also receives the constructs created by your {doc}`subparsers`. If you select a custom device,
the handler registered on `omp_construct` is used **only if you delegate**: `omp_construct(ctr,
block, ctx)` dispatches on the class, so delegating is how a custom device reaches it.
