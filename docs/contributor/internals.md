# Internal Modules

Reference generated from the docstrings of the internal modules, organised like the repository
({doc}`index`). These interfaces can change between releases; the supported extension points are
documented in {doc}`extensions/reference`.

```{contents}
:local:
:depth: 2
```

## Options

```{eval-rst}
.. autoclass:: omp4py.core.api.OmpType
   :special-members: __init__, __call__

.. autoclass:: omp4py.core.api.OmpKwargs

.. automodule:: omp4py.core.options
   :members:
```

## Import hooks

```{eval-rst}
.. automodule:: omp4py.core.imports.loader
   :members:
```

## Preprocessor

```{eval-rst}
.. automodule:: omp4py.core.preprocessor.preprocessor
   :members:

.. automodule:: omp4py.core.preprocessor.obj2ast
   :members:

.. automodule:: omp4py.core.preprocessor.cbuild
   :members:
```

## Transformers

```{eval-rst}
.. automodule:: omp4py.core.preprocessor.transformers.transformer
   :members:

.. automodule:: omp4py.core.preprocessor.transformers.context
   :members:

.. automodule:: omp4py.core.preprocessor.transformers.symtable
   :members:

.. automodule:: omp4py.core.preprocessor.transformers.scopes
   :members:

.. automodule:: omp4py.core.preprocessor.transformers.operators
   :members:

.. automodule:: omp4py.core.preprocessor.transformers.utils
   :members:
```

The constructs themselves are registered by these modules (they export no names, so only the
module docstrings are shown):

```{eval-rst}
.. automodule:: omp4py.core.preprocessor.transformers.parallelism
.. automodule:: omp4py.core.preprocessor.transformers.worksharing
   :members: for_range_check, for_checks, section_checks, copyprivate, check_nowait
.. automodule:: omp4py.core.preprocessor.transformers.mastersync
.. automodule:: omp4py.core.preprocessor.transformers.dataenv
.. automodule:: omp4py.core.preprocessor.transformers.reduction
```

## Modifiers

```{eval-rst}
.. automodule:: omp4py.core.modifiers.engine
   :members:
   :no-index:

.. automodule:: omp4py.core.modifiers.omp.typing
```

## Directive parser

```{eval-rst}
.. automodule:: omp4py.core.parser.parser
   :members:

.. automodule:: omp4py.core.parser.clause_assignment
   :members:

.. automodule:: omp4py.core.parser.error
   :members:
```

### Parse tree nodes

```{eval-rst}
.. automodule:: omp4py.core.parser.tree
   :members:
```

## Runtime: tasks

```{eval-rst}
.. automodule:: omp4py.runtime.tasks.context
   :members:

.. automodule:: omp4py.runtime.tasks.task
   :members:

.. automodule:: omp4py.runtime.tasks.parallelism
   :members:

.. automodule:: omp4py.runtime.tasks.worksharing
   :members:

.. automodule:: omp4py.runtime.tasks.mastersync
   :members:

.. automodule:: omp4py.runtime.tasks.privatization
   :members:

.. automodule:: omp4py.runtime.tasks.threadprivate
   :members:

.. automodule:: omp4py.runtime.tasks.typing
   :members:
```

## Runtime: ICVs

```{eval-rst}
.. automodule:: omp4py.runtime.icvs.icvs
   :members:

.. automodule:: omp4py.runtime.icvs.places
   :members: parse
```

## Runtime: low level

```{eval-rst}
.. automodule:: omp4py.runtime.lowlevel.atomic
   :members:

.. automodule:: omp4py.runtime.lowlevel.mutex
   :members:

.. automodule:: omp4py.runtime.lowlevel.numeric
   :members:

.. automodule:: omp4py.runtime.lowlevel.threadlocal
   :members:
```
