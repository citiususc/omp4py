# The Extensions Module

OMP4Py can be extended **without modifying its source tree**. The module
{py:mod}`omp4py.core.ext` is the public interface that independent projects use to plug into the
preprocessor. It has three extension points:

````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} Subparsers
:link: subparsers
:link-type: doc

Add **new directives**: your own syntax, parsed by hand into a new tree node.
```

```{grid-item-card} Devices
:link: devices
:link-type: doc

Provide **your own transformations** for the constructs, or override some of the default ones.
```

```{grid-item-card} Modifiers
:link: modifiers
:link-type: doc

Add **AST passes** before or after the transformer: optimisations, types, clean-up.
```
````

```{toctree}
:hidden:
:maxdepth: 1

subparsers
devices
modifiers
combining
reference
```

| Extension point | Register with | Enable in `omp(...)` with |
| --- | --- | --- |
| Subparser | `@add_parser("name")` | `parsers=["name"]` |
| Device | `@add_device("name")` | `device="name"` |
| Modifier | `@add_modifier` | `modifiers={"name": True}` |

Two more tools are re-exported by the module and used inside extensions:
{py:data}`~omp4py.core.ext.omp_construct` (the default `omp` transformer, used to register
handlers for new nodes and to delegate) and {py:func}`~omp4py.core.ext.syntax_error_ctx` (raise
a located `SyntaxError`).

## How the pieces fit

```{mermaid}
flowchart TB
    subgraph user["User code"]
        U1["omp(extension='my_ext')"]
        U2["@omp(device='mydev', parsers=['mysub'], modifiers={'mypass': True})"]
        U3["with omp('parallel ...'):   or   mysub('...')"]
    end
    subgraph ext["Extension module (my_ext.py)"]
        E1["@add_parser('mysub')<br/>text → tree node"]
        E2["@omp_construct.register<br/>tree node → Python AST (default device)"]
        E3["@add_device('mydev')<br/>your transformations, delegate the rest to omp"]
        E4["@add_modifier<br/>AST pass"]
    end
    U1 -. "lazy import when the<br/>preprocessor first runs" .-> ext
    U3 --> P{"which parser?"}
    P -- "omp(...)" --> BP["built-in parser<br/>→ Parallel, For, ..."]
    P -- "mysub(...)" --> E1
    BP --> D{"device"}
    E1 --> D
    D -- "'omp'" --> E2
    D -- "'mydev'" --> E3
    E2 --> M["modifiers"]
    E3 --> M
    E4 --> M
```

## Registering and loading an extension (lazily)

An extension is an importable Python module that uses the decorators above. The user
registers it **by name**:

```python
from omp4py import omp

omp(extension="my_ext")        # only records the name: nothing is imported yet
```

The module is imported later, **the first time the preprocessor runs**
(`load_extensions()`), and only once.

The laziness is deliberate. The preprocessor (parser, transformers, modifiers, Cython build
support) is made of many modules, and a program that only *runs* code that was already processed,
or that never uses directives, should not pay for importing them. An extension imports
`omp4py.core.*`, which would load the preprocessor; by registering it by name, that cost is paid
only when processing is actually needed.

The recommended layout for a distributable extension:

```text
my_ext/
├── __init__.py       # lightweight: only registers the implementation
└── _impl.py          # imports omp4py.core.ext, defines parsers, devices, modifiers
```

```python
# my_ext/__init__.py
from omp4py import omp

omp(extension="my_ext._impl")      # lazy: _impl is imported by the preprocessor when needed
```

Users then write `import my_ext` (cheap) and use the extension in their `@omp(...)` options.

Rules:

1. Call `omp(extension=...)` **before** the first decorated function that needs the extension.
2. The module must be importable under the registered name.
3. After editing an extension, run with `ignore_cache=True` (or `OMP4PY_IGNORE_CACHE=1`): the cache
   key is the source file that defines your function, not the extension's.

## When to extend and when to modify the core

| Goal | Use |
| --- | --- |
| A new directive that OpenMP does not define | A **subparser** (+ a handler on `omp_construct`) |
| Generate different code for existing constructs (tracing, serial, instrumented, another back-end) | A **device** |
| Post-process or prepare the code (optimise, add types, strip things) | A **modifier** |
| Implement a missing *standard* OpenMP construct for everybody | The core: {doc}`/contributor/adding_directive` |
