# User Guide

Everything you need to write, run and tune OMP4Py programs. Read it in order the first time;
afterwards use it as a reference.

````{grid} 1 2 2 3
:gutter: 3

```{grid-item-card} The omp decorator
:link: decorator/index
:link-type: doc

The entry point for everything: directives, functions, packages, files, aliases and all its
arguments. **Start here.**
```

```{grid-item-card} Directives
:link: directives
:link-type: doc

`parallel`, `for`, `sections`, `single`, `critical`, `barrier`, ... each with a runnable example.
```

```{grid-item-card} Combined directives
:link: combined_directives
:link-type: doc

`parallel for` and the other OpenMP 6.0 combinations, and clause modifiers.
```

```{grid-item-card} Clauses
:link: clauses
:link-type: doc

Data sharing, control and scheduling: what each clause does.
```

```{grid-item-card} Runtime routines
:link: runtime_api
:link-type: doc

The `omp_*` functions of the OpenMP 4.5 API that you import with `omp`.
```

```{grid-item-card} Environment variables
:link: environment
:link-type: doc

`OMP_*` variables, `OMP_DISPLAY_ENV` and the `OMP4PY_*` settings.
```

```{grid-item-card} Execution modes
:link: execution_modes
:link-type: doc

Pure, hybrid, compiled and compiled with types.
```

```{grid-item-card} Compiled mode
:link: compiled_mode
:link-type: doc

Function vs module compilation, why types matter, and the HTML report.
```

```{grid-item-card} Performance
:link: performance
:link-type: doc

What OMP4Py does and does not speed up, the GIL, and OpenMP good practices.
```

```{grid-item-card} Debugging
:link: debugging
:link-type: doc

Generated code, error messages and pitfalls.
```
````

## Suggested reading order

```{mermaid}
flowchart LR
    A["The omp decorator"] --> B["Directives"] --> C["Clauses"] --> D["Runtime routines"]
    D --> E["Execution modes"] --> F["Compiled mode"] --> G["Performance"]
    B --> H["Combined directives"]
    G --> I["Debugging"]
```

```{important}
This guide documents what OMP4Py implements today. The OpenMP 4.5 coverage, including what is
still in development, is summarised in {doc}`/compatibility/coverage`.
```
