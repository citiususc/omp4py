# Combining Devices and Subparsers

Subparsers and devices are independent but are designed to work together: a subparser
creates constructs, and a device (or the default transformer) decides what they become.
This page walks through a complete backend that uses both.

## Scenario: a profiling backend

We want:

- a **new construct**, `timed("label")`, to time any block of code;
- a **new device**, `profile`, that *also* times every `parallel` region of the program,
  without the user adding anything.

```{mermaid}
flowchart TB
    subgraph user["User code"]
        W["@omp(device='profile', parsers=['timed'])"]
        T["with timed('whole function'):"]
        P["with omp('parallel for ...'):"]
    end
    T -->|"timed(...)"| SP["subparser: parse_timed<br/>→ Directive[Timed]"]
    P -->|"omp(...)"| BP["built-in parser<br/>→ Directive[Parallel, For]"]
    SP --> DV
    BP --> DV
    subgraph DV["device 'profile'"]
        direction TB
        M{"construct?"}
        M -- Timed --> A["wrap block with timing code"]
        M -- Parallel --> B["omp_construct(...) + timing around the result"]
        M -- other --> C["omp_construct(...) unchanged"]
    end
```

## The extension

```{literalinclude} ../../examples/ext/profile_ext.py
:language: python
```

Key points:

- **`Timed`** is a dataclass construct, created by the subparser `parse_timed`
  (registered with `@add_parser("timed")`).
- **`profile`** is registered with `@add_device("profile")`. Because a custom device receives
  *every* construct (including `Timed`), it must handle the custom one explicitly.
- For `Parallel` it **reuses** the default transformer through `omp_construct(...)` and
  surrounds the statements it returns with timing code. Reuse plus decoration is the
  easiest way to build on the standard behaviour.
- `timing_wrapper` builds AST from source text with `ast.parse` and copies the directive
  location onto the nodes. `omp_name(ctx, "t0")` produces a variable name that cannot clash
  with the user's.

## The user code

```{literalinclude} ../../examples/ext/profile_user.py
:language: python
```

Running it:

```text
parallel region took 0.0128 s
whole function took 0.0129 s
19999900000
```

(The timings differ on every run.) The first line comes from the device (it wrapped the
`parallel` region), the second from the `timed` construct.

## Decisions to take when designing a backend

### Where does each custom construct get its meaning?

| If the user selects... | Custom construct handled by |
| --- | --- |
| Default device (`"omp"`) | A handler registered with `@omp_construct.register`. |
| A custom device | The device function itself (default handlers are not consulted). |

To make a construct work in **both**, register a handler with `omp_construct.register` and let
your device delegate unknown constructs to `omp_construct(...)`: custom constructs fall through
to their registered handler, and standard constructs to the standard transformer.

```python
@omp_construct.register
def _(ctr: Timed, block, ctx):
    return timing_wrapper(ctr.label, [block], ctx, ctr.span)

@add_device("profile")
def profile(ctr, block, ctx):
    if isinstance(ctr, Parallel):
        return timing_wrapper("parallel region", omp_construct(ctr, block, ctx), ctx, ctr.span)
    return omp_construct(ctr, block, ctx)      # Timed is dispatched to its handler
```

### Compound directives

If your subparser returns several constructs, they are processed outside in, exactly like
`parallel for`: after each handler returns, its statements (including the block) are
visited again to apply the remaining constructs.

### Data sharing

A device that moves the user's block into a new function (as the default `parallel` does)
changes variable scopes. If your construct needs `private`/`shared` semantics, reuse
`create_scope` and `check_scopes` from `transformers/scopes.py` (see
{doc}`/contributor/architecture`); otherwise keep the block where it is, as `Timed` does.

### Errors

Use `syntax_error_ctx(message, ctr.span, ctx)` for every user error: it produces the same
`SyntaxError` format (file, line, column) as built-in constructs.

## Packaging the extension

For a distributable extension:

```python
# my_backend/__init__.py
from omp4py import omp

omp(extension="my_backend._impl")      # lazy: _impl imports omp4py.core.*

__all__ = []
```

Users then write:

```python
import my_backend
from omp4py import omp

@omp(device="profile", parsers=["timed"])
def run(): ...
```

or, for a whole package, `omp(pkg=__package__, device="profile", parsers=["timed"])`.
