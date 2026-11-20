# Extension API Reference

The registration functions and the base classes you subclass are documented here. The types
that appear in their signatures (parse tree nodes, `Context`, transformer helpers) are
documented in {doc}`/contributor/internals`.

## `omp4py.core.ext`

```{eval-rst}
.. automodule:: omp4py.core.ext
   :no-members:

.. autofunction:: omp4py.core.ext.add_parser

.. autofunction:: omp4py.core.ext.add_device

.. autofunction:: omp4py.core.ext.add_modifier

.. autodata:: omp4py.core.ext.omp_construct
   :no-value:

.. autofunction:: omp4py.core.ext.syntax_error_ctx
```

## Modifiers

```{eval-rst}
.. autoclass:: omp4py.core.modifiers.engine.Modifier
   :members: should_run, run
   :special-members: __init_subclass__

.. autoclass:: omp4py.core.modifiers.engine.ModifierEngine
   :members: stage
```

## Types used by extensions

| Name | Defined in | Role |
| --- | --- | --- |
| {py:class}`~omp4py.core.parser.tree.Span` | `omp4py.core.parser.tree` | Source location; `span.to_ast(node)` copies it onto generated AST nodes. |
| {py:class}`~omp4py.core.parser.tree.Directive` | `omp4py.core.parser.tree` | What a parser returns: `Directive(span, string, constructs)`. |
| {py:class}`~omp4py.core.parser.tree.Construct` | `omp4py.core.parser.tree` | Base class of constructs, including your own. |
| {py:class}`~omp4py.core.parser.tree.Clause` | `omp4py.core.parser.tree` | Base class of clauses. |
| {py:class}`~omp4py.core.parser.tree.Name` | `omp4py.core.parser.tree` | A name with its location (`Name(span, "trace")`). |
| {py:class}`~omp4py.core.preprocessor.transformers.context.Context` | `omp4py.core.preprocessor.transformers.context` | Per-module transformation state handed to every transformation. |

## Transformer helpers

| Function | Defined in |
| --- | --- |
| {py:func}`~omp4py.core.preprocessor.transformers.transformer.with_block` | `omp4py.core.preprocessor.transformers.transformer` |
| {py:func}`~omp4py.core.preprocessor.transformers.transformer.check_control_flow` | `omp4py.core.preprocessor.transformers.transformer` |
| {py:func}`~omp4py.core.preprocessor.transformers.transformer.transform` | `omp4py.core.preprocessor.transformers.transformer` |
| {py:func}`~omp4py.core.preprocessor.transformers.symtable.omp_name` | `omp4py.core.preprocessor.transformers.symtable` |
| {py:func}`~omp4py.core.preprocessor.transformers.symtable.runtime_ast` | `omp4py.core.preprocessor.transformers.symtable` |

## Registries

| Name | Type |
| --- | --- |
| `omp4py.core.preprocessor.transformers.transformer.DEVICES` | `dict[str, Device]` |
| `omp4py.core.preprocessor.transformers.transformer.PARSERS` | `dict[str, Parser]` |
| `omp4py.core.modifiers.engine.MODIFIERS` | `dict[str, type[Modifier]]` |

## Type aliases

```python
type Parser = Callable[[str, Span, str], Directive]
type Device = Callable[[Construct, StructuredBlock | None, Context], list[ast.stmt]]
type StructuredBlock = ast.With | ast.If | ast.Try
```
