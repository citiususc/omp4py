# Subparsers

A **subparser** adds a **new directive**. It is a function that receives the text of the directive,
parses it **by hand**, and returns a node of the directive tree. That node then has to be
processed by a transformer.

Two steps, in two different stages of the pipeline:

```{mermaid}
flowchart LR
    T["text of the directive<br/><code>greet('world')</code>"] -->|"step 1 · subparser<br/>@add_parser"| N["tree node<br/><code>Greet(word='world')</code>"]
    N -->|"step 2 · transformer<br/>@omp_construct.register"| A["Python AST<br/><code>print('hello, world')</code>"]
```

## Step 1: the subparser creates a new node

The subparser gets the text, the source location and the file name, and returns a
{py:class}`~omp4py.core.parser.tree.Directive` that holds one or more *constructs*. A construct is
a dataclass that derives from {py:class}`~omp4py.core.parser.tree.Construct`: that is the "new tree
node".

```python
from dataclasses import dataclass
from omp4py.core.ext import add_parser
from omp4py.core.parser import Construct, Directive, Name, Span


@dataclass
class Greet(Construct):                # the new tree node
    word: str = ""


@add_parser("greet")                   # the name of the subparser
def parse_greet(code: str, span: Span, filename: str) -> Directive:
    word = code.strip()                # the whole directive is one word: "parsed by hand"
    return Directive(span, code, [Greet(span, Name(span, "greet"), word=word)])
```

- `code` is the text of the string literal, surrounded by padding that keeps the original column
  positions (`strip()` it).
- `span` is the source location; give it to every node you build.
- To reject invalid input, raise `SyntaxError`.

Users enable the subparser through the `parsers` argument, and call it **by its name**, exactly
like `omp(...)`:

```python
@omp(parsers=["greet"])
def main():
    greet("world")
```

## Step 2: the transformer must process the new node

The node `Greet` exists now, but nothing knows what to do with it. **Without a handler the
preprocessor stops with an error**:

```text
SyntaxError: 'greet' not implemented
```

The default transformer is the `omp` device, a `functools.singledispatch` function keyed by the
**class** of the construct. Register a handler for your class on `omp_construct`:

```python
import ast
from omp4py.core.ext import omp_construct

@omp_construct.register
def _(ctr: Greet, block, ctx):
    node = ast.Expr(ast.Call(ast.Name("print"), [ast.Constant(f"hello, {ctr.word}")], []))
    return [ast.fix_missing_locations(ctr.span.to_ast(node))]
```

A handler receives the construct, the structured `block` it applies to (`None` for a stand-alone
statement) and the context. It returns the list of statements that replace the directive. **If
there is a block, it must be in the list**, so that nested directives and the remaining constructs of
the same directive can be processed.

## The complete example

```{literalinclude} ../../examples/ext/greet_ext.py
:language: python
```

```{literalinclude} ../../examples/ext/greet_user.py
:language: python
```

```text
hello, world
hello, OMP4Py
```

The call `greet("world")` is replaced during preprocessing, so `greet` does not need to exist at
run time. Linters do not know that: add `# noqa: F821` or define a stub.

## A directive with a block

A construct can wrap code. Return the block between your own statements:

```python
@omp_construct.register
def _(ctr: Trace, block, ctx):
    if block is None:
        raise syntax_error_ctx("trace must be used in a 'with' block", ctr.span, ctx)
    return [print_enter, block, print_exit]
```

(`syntax_error_ctx` is the helper of `omp4py.core.ext` that raises a `SyntaxError` positioned at the
directive.) A complete block-wrapping example is in {doc}`combining`.

## Parsing real syntax

For anything richer than a word, parse the text yourself and keep the parser a plain function
(text in, nodes out), so that it is easy to test:

```python
import re

@add_parser("kernel")
def parse_kernel(code, span, filename):
    m = re.fullmatch(r"\s*(?P<name>\w+)\s*(?:\(\s*(?P<n>\d+)\s*\))?\s*", code)
    if m is None:
        raise SyntaxError(f"invalid kernel directive: {code!r}")
    return Directive(span, code, [Kernel(span, Name(span, "kernel"),
                                         kernel=m["name"], size=int(m["n"] or 1))])
```

A directive may return several constructs; they are processed from the outermost to the innermost,
like `parallel for`.

## Subparsers and devices

The handler registered on `omp_construct` is used by the default device. If the user selects
**another device**, every construct, including yours, is sent to that device, which must handle it
or delegate to `omp_construct`. See {doc}`devices` and {doc}`combining`.

Both options propagate to packages:

```python
omp(pkg=__package__, parsers=["greet"])
```
