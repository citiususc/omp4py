# Modifiers

A **modifier** is a pass over the whole module (`ast.Module`) that runs **before or after** the
transformer. Use it to add optimisations or prepare/clean the code **without redefining what the
directives mean**.

```{mermaid}
flowchart LR
    S["source AST"] --> B["modifiers<br/><b>before</b> the transformer<br/><code>requires=[]</code>"]
    B --> T["transformer<br/>(generates the code)"]
    T --> A["modifiers<br/><b>after</b> the transformer<br/><code>requires=['omp_transformer']</code> (default)"]
    A --> O["final code"]
```

## The class

```python
from omp4py.core.ext import Modifier, add_modifier

@add_modifier
class MyPass(Modifier, name="my_pass", requires=[], default=False):
    def visit_Call(self, node): ...
```

`Modifier` is an `ast.NodeTransformer`: implement the usual `visit_*` methods. The class statement
takes keyword arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `name` | *required* | Identifier used to enable it |
| `requires` | `["omp_transformer"]` | What must have run before it. `["omp_transformer"]` means **after the transformer**; no requirements (`[]`) means **before** it; the name of another modifier means after that one |
| `default` | `False` | Whether it runs when the user does not mention it |

Each instance has `self.options` (the `Options`) and `self.metadata`, a dictionary shared by all the
modifiers of one run, to pass information between passes. Override the class method
`should_run(cls, options)` for extra conditions.

## Enable them with `omp(...)`

A modifier only runs when it is enabled, either because its `default` is `True` or because the user
asks for it in the `modifiers` argument of **`omp`**:

```python
@omp(modifiers={"my_pass": True})
def f(): ...

omp(pkg=__package__, modifiers={"my_pass": True})
```

`modifiers_args={"my_pass": {...}}` passes free-form configuration, available as
`self.options.modifiers_args`.

## Example 1: after the transformer

A modifier that removes `assert` statements from the final code, a typical
"remove what is not needed in production" pass:

```{literalinclude} ../../examples/ext/modifier_ext.py
:language: python
:lines: 1-14
```

```{literalinclude} ../../examples/ext/modifier_user.py
:language: python
:lines: 1-12
```

```text
-5
```

The assertion would have raised for `-5`. Returning `None` from a `visit_*` method deletes the node.

## Example 2: before the transformer

A modifier with **no requirements** runs on the source AST, *before* the transformer reads the
directives. That lets it prepare code that the transformer would otherwise reject. A
worksharing `for` only accepts `range(...)` loops; this modifier accepts the Python 2 spelling:

```{literalinclude} ../../examples/ext/modifier_ext.py
:language: python
:lines: 16-
```

```{literalinclude} ../../examples/ext/modifier_user.py
:language: python
:lines: 15-
```

```text
[0, 1, 4, 9, 16]
```

Had the modifier run *after* the transformer, it would be too late: the loop would already have
been rejected with `invalid controlling predicate`.

## The modifiers that OMP4Py itself uses

Both live in `omp4py/core/modifiers/omp/typing.py`, are enabled by default and decide by
themselves whether to run (`should_run`), so users do not normally touch them:

| Modifier | Runs when | What it does |
| --- | --- | --- |
| `omp_cython_types` | `compile=True` with the Cython compiler | Rewrites the generic typing helpers emitted by the transformer (`_omp.tp_cast`, `_omp.tp_typeof`) into `cython.cast` / `cython.typeof`, and adds the `cython` import, so the generated code benefits from static typing |
| `omp_pure_types` | Without compilation (`compile=False`) | Simplifies or removes those helpers, which are useless without Cython, to avoid overhead |

Both are examples of the intended use of modifiers: **adding optimisations** that depend on how the
code will be executed. To switch one off, use `omp(modifiers={"omp_pure_types": False})`.

There is also a modifier that is **off by default** and that you enable with
`omp(modifiers={"omp_local_aliases": True})` (`omp4py/core/modifiers/omp/localvars.py`). In every
function nested in another one, such as the function generated for a `parallel` region, it binds a local name
to each variable that the function reads and does not assign (`_omp_1w = w`), using the renaming of the symbol
table of the transformer, and reads that name instead. It is off by default because binding the variable when
the function starts changes what the program does if some code that the function calls assigns that variable
while it runs. It does not run when the code is compiled.

## Typical uses

- Optimisations specific to a back-end (types, casts, inlining).
- Removing code that is not needed in a given mode (debug-only code, assertions, logging).
- Instrumentation (timing, counters) around the generated code.
- Preparing source for the transformer (accepting extra syntax, as above).
