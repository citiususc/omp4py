# Architecture

## The preprocessing pipeline

`omp(...)` is a thin front door (`omp4py.core.api.OmpType`). The real work happens in
{py:mod}`omp4py.core.preprocessor.preprocessor`.

```{mermaid}
flowchart TB
    subgraph entry["Entry points (core/api.py)"]
        E1["@omp on function/class"]
        E2["omp(pkg=...)"]
        E3["omp(py=...)"]
    end
    E1 --> PO["process_object()"]
    E2 --> LD["Omp4pyFinder / Omp4pyLoader<br/>(imports/loader.py)"]
    E3 --> PF["process_file()"]
    LD --> PS["process_source()"]
    PF --> PS
    PO --> PR["process()"]
    PS --> PR
    PR --> X["load_extensions()<br/>(extensions register subparsers,<br/>devices and modifiers)"]
    X --> M1["ModifierEngine.stage(module)<br/>pre-transform modifiers"]
    M1 --> TR["OmpTransformer.transform()"]
    TR --> M2["ModifierEngine.stage(module, 'omp_transformer')"]
    M2 --> OUT{"opt.compile?"}
    OUT -- no --> PYOUT["ast.unparse → cached .py → exec"]
    OUT -- yes --> CY["cythonize() → extension module"]
```

`process()` in `preprocessor/preprocessor.py` is the one function every path goes
through. It:

1. imports any registered extension modules (`load_extensions()`),
2. creates an `OmpTransformer` and a `ModifierEngine`,
3. runs the modifiers that do not depend on the transformer,
4. runs the transformer,
5. runs the modifiers that depend on the `omp_transformer` stage,
6. optionally writes the result to `opt.dump`.

## The transformer: two passes

`OmpTransformer.transform()` (`preprocessor/transformers/transformer.py`) walks the AST
twice.

**Pass 1: `OmpParser` (an `ast.NodeVisitor`)** finds every call to the alias (`omp`) or to
a **subparser** name enabled in `parsers`, validates its shape (one constant-string argument, used as a
statement or as a single-item `with`), and parses the string into a
{py:class}`~omp4py.core.parser.tree.Directive`. The result is stored in
`ctx.directives`, keyed by the `ast.With` / `ast.Expr` node. Decorators with the alias are
removed here (`find_decorator`).

**Pass 2: `OmpTransformer` (an `ast.NodeTransformer`)** visits the tree again. For every
node that owns a directive it takes the directive's constructs one by one and calls
`transform(construct, block, ctx)`:

```{mermaid}
flowchart TB
    V["visit_With / visit_Expr"] --> Q{"constructs left<br/>for this node?"}
    Q -- no --> BODY["replace node by its body"]
    Q -- yes --> POP["pop next construct"]
    POP --> T["transform(construct, block, ctx)"]
    T --> D{"ctx.opt.device"}
    D -- "'omp' (default)" --> C["construct(...)<br/>singledispatch by construct class"]
    D -- "other name" --> DV["a device registered<br/>by an extension<br/>(looked up in DEVICES)"]
    C --> R["list of new statements<br/>(must include the block)"]
    DV --> R
    R --> REV["generic_visit the result:<br/>inner nodes and remaining constructs"]
    REV --> V
```

Important consequences:

- A *compound* directive such as `parallel for` is a list of constructs
  `[Parallel, For]` applied **outside in**: `Parallel` wraps the block in a function,
  the block comes back in the result, and `For` is applied to it on the next visit.
- A transformation that has a block **must return it** among its statements; otherwise
  the following constructs have nothing to work on and the user's code disappears.
- When nested `with omp(...)` statements form a chain (`with omp("parallel"): with
  omp("for"):`), they are merged into one compound sequence.

## The `Context`

`Context` (in `transformers/context.py`) is the object every transformation receives.

| Field | Meaning |
| --- | --- |
| `full_source`, `module`, `opt` | Source text, root AST, active {py:class}`~omp4py.core.options.Options`. |
| `scope` | Current `Scope`: AST node, `SymbolTable`, name-generation counters. |
| `module_storage` | Per-module state shared between decorated functions: declared reductions, threadprivate names, named mutexes. |
| `node_stack` | Ancestors of the node being visited. |
| `directives`, `active_constructs`, `used_constructs` | Pass-1 results and pass-2 bookkeeping. |
| `finalizers`, `node_finalizers` | Callbacks run at the end of the module / of the current node (used, for example, to hoist a named mutex to the top of the module). |

## Symbols and scopes

`symtable.py` builds on Python's own `symtable` module to know, for every name in
a function, whether it is global, local, free, assigned, threadprivate, and so on.
`scopes.py` uses that information to implement data sharing:

- `check_scopes` validates the names of a clause (defined, not duplicated, not
  predetermined threadprivate).
- `create_scope` renames private variables (`_omp_1total`), emits their initialisation,
  and inserts `nonlocal`/`global` declarations so shared variables written inside the
  generated function still refer to the original.
- `omp_name(ctx, "parallel")` produces unique generated identifiers.
- `runtime_ast("parallel")` builds the AST node `_omp.parallel`.

## The directive parser

```{mermaid}
flowchart LR
    S["string constant<br/>in the AST"] -->|extract_directive| T["directive text<br/>(with original positions)"]
    T -->|"call to omp(...)<br/>openmp.lark → Lark LALR"| P["parse tree"]
    P -->|OpenmpTransformer| D["Directive<br/>constructs + clauses"]
    T -->|"call to a subparser name<br/>(function found in PARSERS)"| D
    D --> CA["ClauseAssignment<br/>(compound directives)"]
```

Files in `omp4py/core/parser/`:

| File | Role |
| --- | --- |
| `string.lark` / `string_lark.py` | Mini-grammar used to recover the exact text of a (possibly multi-line, escaped) string literal so error carets land on the right column. |
| `openmp.lark` / `openmp_lark.py` | The directive and clause grammar. `openmp_lark.py` is **generated** (standalone Lark parser). |
| `compound.lark`, `compound_gen.py`, `compound.py` | Valid compound directive names (OpenMP 6.0, appendix D). `compound.py` is generated into the `DIRECTIVES` set. |
| `parser.py` | `extract_directive`, `parse_directive` and the `OpenmpTransformer` that turns the Lark tree into dataclasses. |
| `tree.py` | The node classes: `Directive`, `Construct` subclasses (`Parallel`, `For`, ...), `Clause` subclasses (`NumThreads`, `Schedule`, ...), modifiers. |
| `clause_assignment.py` | Decides which construct of a compound directive receives each clause (`private` → innermost, `nowait` → outermost, ...). |
| `error.py` | Converts Lark errors into `SyntaxError`s with positions mapped back to the user's file. |

Every node derives from `OmpNode`, which carries a {py:class}`~omp4py.core.parser.tree.Span`
(line/column in the user's source) so that any later stage can raise a precisely
located `SyntaxError` through `syntax_error_ctx`.

## Modifiers

Modifiers (`core/modifiers`) are small `ast.NodeTransformer` passes selected by name in
`omp(modifiers={...})`. The only built-in one, `omp_cython_types`, rewrites the generic
typing helpers emitted by the transformer (`_omp.tp_cast`, `_omp.tp_typeof`) into
`cython.cast`/`cython.typeof` when compiling. See {doc}`extensions/modifiers`.

## Where generated code comes from

A transformer builds AST nodes directly. The calls it emits (`_omp.parallel`,
`_omp.for_init`, `_omp.critical_lock`, ...) are the public surface of
`omp4py.runtime` listed in its `__init__`. The module is imported into the user's
namespace under the name `_omp` by the preprocessor, so generated code stays readable
(see the dump in {doc}`/user_guide/debugging`).
