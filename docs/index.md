# OMP4Py Documentation

## A native Python implementation of OpenMP

```{image} _static/logo.png
:alt: logo
:class: bg-primary
:width: 30%
:align: left
```

**OMP4Py** is a Python library that provides a native implementation of OpenMP, 
which is widely recognized as the standard programming model for exploiting 
multithreading parallelism in HPC. OMP4Py enables developers to express 
familiar OpenMP constructs directly in Python through the {func}`omp4py.omp`
entry point, which serves as the unified interface to manage directives, 
decorators and module registrations across the entire system.

OMP4Py features a dual-runtime architecture, consisting of a pure Python 
runtime and a native C-based runtime generated using Cython. This design 
combines the flexibility and ease of use of Python with the performance 
benefits of native execution, enabling developers to write parallel code 
with the same level of control and efficiency as in C, C++, or Fortran.

Compared to the Numba-based [PyOMP](https://github.com/Python-for-HPC/PyOMP),
OMP4Py offers greater flexibility, as it imposes no restrictions on using 
functions from non-Numba-optimized libraries or Python objects. Experimental
results show that OMP4Py achieves good performance and scalability for both 
numerical and non-numerical tasks. Additionally, OMP4Py code can be combined 
with [mpi4py](https://github.com/mpi4py/mpi4py) to develop parallel 
applications that exploit both intra-node and inter-node parallelism.

The documentation is organized for two audiences:

- **Final users** who want to install OMP4Py, write parallel Python programs,
  run examples, and understand the supported directives.
- **Developers** who want to understand the architecture, modify or extend 
  OMP4Py's core capabilities, run tests, or build and publish this 
  documentation.

## Contents

```{toctree}
:caption: Getting Started
:maxdepth: 2

getting_started/index
getting_started/installation
getting_started/quickstart
```

## Citing OMP4Py

If you use OMP4Py in academic work, please cite:

- César Piñeiro and Juan C. Pichel. [Unlocking Python Multithreading 
  Capabilities using OpenMP-Based Programming with OMP4Py](
  https://doi.org/10.1109/CGO68049.2026.11395224). IEEE/ACM International
  Symposium on Code Generation and Optimization (CGO), 2026.
- César Piñeiro and Juan C. Pichel. [OMP4Py: A pure Python implementation
  of OpenMP](https://doi.org/10.1016/j.future.2025.108035). Future Generation
  Computer Systems, Vol. 175, 2026.


