# Contributor Guide

For people who want to understand OMP4Py from the inside, fix bugs, complete OpenMP support, or
**extend** the tool without touching its core.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Page
  - Contents
* - {doc}`architecture`
  - The big picture: the preprocessing pipeline, the transformer, the context and the directive parser.
* - {doc}`runtime`
  - How the runtime works, how pure and compiled versions share one source, and how to build it.
* - {doc}`loaders`
  - The import hooks that process and compile packages on the fly, and the `__omp__` folder.
* - {doc}`adding_directive`
  - Recipe to add a construct to OMP4Py itself, from the grammar to the runtime.
* - {doc}`extensions/index`
  - **The extension API**: add behaviour from an independent project with subparsers, devices and modifiers.
* - {doc}`code_quality`
  - Ruff and Ty.
* - {doc}`internals`
  - Generated reference of the internal modules.
```

## Repository map

```text
omp4py/
├── __init__.py            public entry: omp + the OpenMP runtime API
├── core/                  the preprocessor (compile-time side)
│   ├── api.py             omp(...) overloads, OmpType
│   ├── options.py         Options dataclass (+ OMP4PY_* env vars)
│   ├── ext.py             extension API (devices, parsers, modifiers)
│   ├── parser/            directive grammar and parse tree
│   ├── preprocessor/      pipeline orchestration + transformers/
│   ├── modifiers/         AST passes (e.g. Cython typing)
│   └── imports/           import hooks, Cython shim, pure-mode importer
└── runtime/               what generated code calls (run-time side)
    ├── api/               omp_get_thread_num(), omp_set_num_threads(), ...
    ├── tasks/             parallel regions, worksharing, barrier, threadprivate
    ├── icvs/              internal control variables, OMP_* parsing
    └── lowlevel/          atomics, mutexes, thread-local storage, numeric aliases
docs/                      this documentation (Sphinx + MyST + Shibuya)
examples/                  benchmark programs (CLI: omp4pyex)
test/                      pytest suite
```

## Two halves: compile time and run time

Everything under `omp4py/core` runs **when your code is processed** (a function is decorated, or
a package module is imported). Everything under `omp4py/runtime` runs **when the program
executes**. The contract between them is the set of names exported by `omp4py.runtime`, which
generated code imports as `_omp`.

```{mermaid}
flowchart LR
    subgraph ct["Compile time: omp4py.core"]
        A["parser"] --> M1["modifiers<br/>(before)"]
        M1 --> B["transformer<br/>(generates the code)"]
        B --> M2["modifiers<br/>(after)"]
        M2 --> CC["compilation<br/>(optional, Cython)"]
    end
    subgraph rt["Run time: omp4py.runtime"]
        D["tasks · icvs · lowlevel"]
    end
    CC -- "generated code calls<br/>_omp.parallel, _omp.for_next_static, ..." --> D
```

The code generation happens **in the transformer**. Modifiers are optional passes that run
before or after it, and compilation only happens when `compile=True`. If you change a
transformer so that it emits a different runtime call, change the runtime too (and the other way
round).

## Extending OMP4Py: devices and subparsers

You rarely need to modify OMP4Py to add behaviour. The extension API lets an **independent
project** plug into the pipeline:

```{mermaid}
flowchart LR
    U["omp('...')"] --> BP["built-in parser<br/>(OpenMP grammar)"]
    U2["mysub('...')"] --> SP["<b>subparser</b><br/>your own syntax"]
    BP --> T["Directive<br/>(constructs + clauses)"]
    SP --> T
    T --> DV{"device"}
    DV -- "omp (default)" --> OT["standard transformations"]
    DV -- "mydev" --> MD["<b>device</b><br/>your transformations"]
    MD -. "delegates the rest" .-> OT
    OT --> MOD["<b>modifiers</b><br/>your AST passes"]
    MD --> MOD
```

| Extension point | What it lets you do | Page |
| --- | --- | --- |
| **Subparser** | Define a new directive (your own syntax) and the *node* that represents it | {doc}`extensions/subparsers` |
| **Device** | Provide your own transformation for each construct (or override some of the default ones), delegating the rest to `omp` | {doc}`extensions/devices` |
| **Modifier** | Add an AST pass before or after the transformer: optimisation, types, clean-up | {doc}`extensions/modifiers` |

Extensions are registered with `omp(extension="module")` and loaded **lazily**, only when the
preprocessor is needed ({doc}`extensions/index`).

## Development setup

OMP4Py uses [uv](https://docs.astral.sh/uv/) to manage the environments and the
[poethepoet](https://poethepoet.natn.io/) task runner. The dependencies are split in **three
`uv` dependency groups**:

| Group | Contains | Use it for |
| --- | --- | --- |
| `dev` | OMP4Py with the `compile` extra, pytest, poethepoet, Lark | Working on the code and running tests |
| `checks` | `dev` + Ruff and Ty | Linting and type checking |
| `docs` | Sphinx, Shibuya, MyST, Mermaid, sphinx-design | Building this documentation |

```bash
git clone https://github.com/citiususc/omp4py.git
cd omp4py
uv sync --reinstall-package omp4py --group dev      # builds the native runtime
uv sync --group checks                              # when you need ruff / ty
uv sync --group docs                                # when you build the docs
```

Choose the interpreter with `-p`, for example a free-threaded one:

```bash
uv run -p 3.14t pytest --pure
```

The `poe` tasks defined in `pyproject.toml`:

| Command | Purpose |
| --- | --- |
| `poe lark` | Regenerate the standalone Lark parsers after editing the `.lark` grammars |
| `poe ruff` | Lint `omp4py/` with Ruff |
| `poe ty` | Type-check with Ty |
| `poe docs` | Build the HTML documentation (`sphinx-build -b html docs html`) |

```bash
uv run --group dev --group docs poe docs
```

`docs/conf.py` sets `OMP4PY_PURE=1` so building the documentation does not need the native runtime.
The API reference pages use `autodoc`, so write Google-style docstrings and they appear in
{doc}`internals` automatically.
