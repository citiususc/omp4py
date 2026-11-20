# Adding a Directive to OMP4Py

This page shows, step by step, how to add a **construct** to OMP4Py *itself*: from the grammar to
the runtime. As a running example we invent a fictitious construct, `stopwatch`, that measures the
time of a block and accepts a `label(...)` clause:

```python
@omp
def work(n):
    total = 0
    with omp("stopwatch label('sum loop')"):
        for i in range(n):
            total += i
    return total
```

```text
[sum loop] 0.042963s
```

```{important}
**When to use this recipe.** It is meant for *improving OMP4Py*: implementing the constructs of the
OpenMP standard. If you only want to add **extra behaviour** to your own programs (new directives,
different code generation, instrumentation), do **not** modify OMP4Py: create an independent
project and use the extension API of devices, subparsers and modifiers, which does not require
touching the core. See {doc}`extensions/index`.
```

Every code block below was applied to a copy of the repository and run: the result is the
output shown above.

```{mermaid}
flowchart LR
    G["1 · Grammar<br/>openmp.lark"] --> T["2 · Tree nodes<br/>tree.py"]
    T --> P["3 · Parser glue<br/>parser.py"]
    P --> X["4 · Transformer<br/>transformers/"]
    X --> R["5 · Runtime<br/>runtime/tasks/"]
    R --> Q["6 · Tests"]
    Q --> D["7 · Documentation"]
```

## 1 · Grammar

The directive grammar is `omp4py/core/parser/openmp.lark`. Add the construct name to the list of
basic constructs and the new clause to the list of clauses:

```text
?basic_construct: lit{"taskyield"i}  | lit{"barrier"i} | ...
                | lit{"master"i}     | lit{"section"i}     | lit{"stopwatch"i}
                ...

clause : ...
       | lit{"label"i}   "(" [_HM list{mod_default} ":"] py_expr   ")"
       | lit{"link"i}    ...
```

`py_expr` accepts a Python expression, and `[_HM list{mod_default} ":"]` lets the clause take a
*directive-name modifier* ({doc}`/user_guide/combined_directives`). Then regenerate the standalone
parser, which is the file the code actually imports:

```bash
poe lark
```

```{warning}
The `*_lark.py` files are **generated**; never edit them by hand. Commit the regenerated file
together with the `.lark` change.
```

## 2 · Tree nodes

Add a dataclass for the construct and one for the clause in `omp4py/core/parser/tree.py`, and
export them in `__all__`:

```python
@dataclass
class Stopwatch(Construct):
    """Represent a stopwatch construct.

    Attributes:
        label (Label | None): Label clause.
    """

    label: Label | None = None


@dataclass
class Label(Clause):
    """Represent a label clause.

    Attributes:
        expr (PyExpr): Expression with the label text.
    """

    expr: PyExpr
```

Naming matters, because the parser connects things **by name**:

- The construct class is found from the directive name: `stopwatch` → `tree.Stopwatch`
  (`basic_directive` capitalises the words).
- A clause is stored in the construct attribute whose name is the clause's identifier:
  class `Label` → attribute `label`; class `NoWait` → `no_wait`; class `If` → `if_`. Use `list[...]`
  for clauses that may repeat (`private`) and `X | None` for single ones. A clause that the
  construct has no attribute for is rejected with `'label' is not valid for 'stopwatch'`.

## 3 · Parser glue

For each new clause, add a `<name>_cl` method to `OpenmpTransformer` in
`omp4py/core/parser/parser.py`. It receives the parsed pieces and builds the node:

```python
def label_cl(self, meta: Meta, name: Name, mods: Modifiers, expr: tree.PyExpr) -> tree.Label:
    return self.set_mods(tree.Label(self.meta2span(meta), name, expr), mods)
```

A construct with the plain syntax `name clauses` needs no extra code: `basic_directive` builds it.
Constructs with special syntax (`critical(name)`, `threadprivate(list)`, `cancel construct`)
have their own rule in the grammar and a `*_dir` method in the same file.

## 4 · Transformer

Create `omp4py/core/preprocessor/transformers/stopwatch.py`, register a handler with
`@construct.register` (the annotation selects the class), and import the module from
`transformers/__init__.py` so that it is registered:

```python
from __future__ import annotations

import ast

from omp4py.core.parser.tree import Stopwatch  # noqa: TC001 required
from omp4py.core.preprocessor.transformers.symtable import runtime_ast
from omp4py.core.preprocessor.transformers.transformer import Context, StructuredBlock, construct, with_block
from omp4py.core.preprocessor.transformers.utils import fix_body_locations

__all__ = []


@construct.register
def _(ctr: Stopwatch, block: StructuredBlock | None, ctx: Context) -> list[ast.stmt]:
    block = with_block(ctr, block, ctx)       # a 'with' block is required; checks control flow
    label: ast.expr = ctr.label.expr.value if ctr.label else ast.Constant("stopwatch")

    start = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("stopwatch_start"))))
    stop = ctr.span.to_ast(ast.Expr(ast.Call(runtime_ast("stopwatch_stop"), [label])))
    return fix_body_locations([start, block, stop])
```

```python
# transformers/__init__.py
import omp4py.core.preprocessor.transformers.stopwatch  #noqa: F401
```

Rules of thumb (taken from the existing handlers):

- **Return the block.** If you receive one, it must be in the returned list, with your extra
  statements *around* it, so later constructs of the same directive can still transform it.
- Use `with_block` when a block is mandatory, `create_scope`/`check_scopes` for data-sharing
  clauses, `check_nowait` for worksharing constructs.
- Copy source positions (`span.to_ast(node)`, `fix_body_locations`) so errors and tracebacks point
  at the user's code, and raise errors with `syntax_error_ctx(message, span, ctx)`.
- `omp_name(ctx, "hint")` creates unique names; `runtime_ast("name")` builds `_omp.name`.

## 5 · Runtime

The generated code calls `_omp.stopwatch_start()` and `_omp.stopwatch_stop(label)`, so they must
exist in the runtime. Create `omp4py/runtime/tasks/stopwatch.py`:

```python
from __future__ import annotations

import time

from omp4py.runtime.tasks.context import omp_ctx

__all__ = ["stopwatch_start", "stopwatch_stop"]

_starts: dict[int, float] = {}


def stopwatch_start() -> None:
    _starts[omp_ctx().icvs.thread_num] = time.perf_counter()


def stopwatch_stop(label: str) -> None:
    elapsed = time.perf_counter() - _starts.pop(omp_ctx().icvs.thread_num)
    print(f"[{label}] {elapsed:.6f}s")
```

and export it from the runtime so generated code can reach it, in `runtime/tasks/__init__.py`:

```python
from omp4py.runtime.tasks.stopwatch import stopwatch_start, stopwatch_stop  # noqa: F401
```

For the **native runtime** ({doc}`runtime`) also:

1. add `omp4py/runtime/tasks/stopwatch.pxd` declaring the functions,
2. cimport them in `runtime/tasks/__init__.pxd` (and `runtime/__init__.pxd` follows it),
3. add `"omp4py/runtime/tasks/stopwatch.py"` to `extensions` in `setup.py`,
4. keep the `.py` code Cython-friendly and consistent with the `.pxd`.

Pure mode needs none of that, which makes it the quickest way to try a new construct.

## 6 · Try it

```python
from omp4py import *

@omp
def per_thread(label):
    with omp("parallel num_threads(2)"):
        with omp("stopwatch label(label)"):
            sum(range(100000))

per_thread("worker")
```

```text
[worker] 0.001421s
[worker] 0.001463s
```

Run it with `OMP4PY_PURE=1 OMP4PY_IGNORE_CACHE=1` the first times, so a cached transformation
from before your change is not reused.

## 7 · Tests and documentation

- Add tests under `test/basic/`: a plain function containing the directive, and a `test_*` that
  preprocesses it with `omp(function)` and checks the result. Add error cases with
  `pytest.raises(SyntaxError)`.
- Document the construct in the {doc}`user guide </user_guide/directives>` with a complete example
  that you have run, and update {doc}`/compatibility/coverage`.

## Checklist

- [ ] Grammar updated and `poe lark` run
- [ ] Tree nodes (`tree.py`, `__all__`) and `*_cl` / `*_dir` parser methods
- [ ] `@construct.register` handler, imported from `transformers/__init__.py`
- [ ] Runtime functions exported in `runtime/tasks/__init__.py` (and `.pxd`, `setup.py` for native)
- [ ] Tests, with and without `--pure`
- [ ] `poe ruff` and `poe ty` clean
- [ ] Documentation and coverage page updated
