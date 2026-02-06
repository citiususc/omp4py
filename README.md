# OMP4Py: a native Python implementation of OpenMP

**OMP4Py** is a Python library that provides an implementation of OpenMP, which is widely recognized as the standard 
programming model for exploiting multithreading parallelism in HPC. OMP4Py features a dual-runtime architecture, 
consisting of a pure Python runtime and a native C-based runtime generated using Cython. This design combines the 
flexibility and ease of use of Python with the performance benefits of native execution, enabling developers to write 
parallel code with the same level of control and efficiency as in C, C++, or Fortran.

Compared to the Numba-based [PyOMP](https://github.com/Python-for-HPC/PyOMP), OMP4Py offers greater flexibility, as it 
imposes no restrictions on using functions from non-Numba-optimized libraries or Python objects. Experimental results 
show that OMP4Py achieves good performance and scalability for both numerical and non-numerical tasks. Additionally, 
OMP4Py code can be combined with [mpi4py](https://github.com/mpi4py/mpi4py) to develop parallel applications that 
exploit both intra-node and inter-node parallelism.

If you use **OMP4Py**, please cite the following works:
* César Piñeiro and Juan C. Pichel. Unlocking Python Multithreading Capabilities using OpenMP-Based Programming with OMP4Py IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 2026.
* César Piñeiro and Juan C. Pichel. [OMP4Py: A pure Python implementation of OpenMP](https://doi.org/10.1016/j.future.2025.108035). Future Generation Computer Systems, Vol. 175, 2026.

## Features

- Native Python library (dual-runtime architecture: pure and native C-based runtime using Cython)
- 4 modes of operation: pure, hybrid, compiled and compiled with data types
- Simplified parallelization of loops and code sections
- Efficient thread management and synchronization
- API compliant with [OpenMP 3.0 standard](https://www.openmp.org/wp-content/uploads/spec30.pdf)

## Installation

You can install omp4py via pip:

```bash
pip install git+https://github.com/citiususc/omp4py.git
```
**Note**: OMP4Py is compatible with Python versions 3.12 and later, which include the Global Interpreter Lock (GIL). 
However, to fully exploit multithreading for scaling applications, it is necessary to use Python 3.13 (free threading) 
or later, which offers a no-GIL option.

## Usage

OMP4Py defines a function `omp` that operates similarly to OpenMP directives in C/C++, maintaining the same syntax and 
functionality. The function itself has no effect when executed; it serves solely as a container for the OpenMP 
directives. Note that when a OpenMP directive must be used within structured blocks, the `omp` function is used together 
as part of a `with` block; otherwise, it is used as a standalone function call. Note that functions or classes 
containing the OpenMP directives must be decorated with the `@omp` decorator.

Here's a basic example of how to use OMP4Py to calculate $\pi$:

```python
    from omp4py import *
    
    @omp
    def pi(n):
        w = 1.0 / n
        pi_value = 0.0
        with omp("parallel for reduction(+:pi_value)"):
            for i in range(n):
                local = (i + 0.5) * w
                pi_value += 4.0 / (1.0 + local * local)
        return pi_value * w

    print(pi(10000000))  
```

OMP4Py can be executed in four different modes:

- **Pure mode**:  
  Executes using the pure Python runtime. To enable it, users must explicitly sets and enviroment variable before 
- importing OMP4Py:  
  ```python
  import os
  os.environ['OMP4PY_PURE'] = 'True'
  from omp4py import *
  ```
  
- **Hybrid mode (default)**:  
This is the standard mode when importing OMP4Py with:
  ```python
  from omp4py import *
  ```
- **Compiled mode**: In this mode, both the runtime and the user’s function are compiled to native code using Cython. 
- This removes Python interpreter overhead and provides improved performance for numerical workloads.

  ```python
  @omp(compile=True)
  ```
  
- **Compiled-with-types mode**: A more optimized version of the previous mode, where the programmer also provides 
- static type annotations, allowing Cython to generate more efficient native code. This can result in substantial 
- speed-ups, up to three orders of magnitude compared with Pure mode.
	```python
	@omp(compile=True)
	def pi(n: int): ...
	```
All versions of the pi implementation using these execution modes can be found in [https://github.com/citiususc/omp4py/blob/main/examples/pi.py](https://github.com/citiususc/omp4py/blob/main/examples/pi.py)

The rest of the examples can be found in the **examples** folder.

You can download OMP4Py and then run with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# In the omp4py directory
uv run -p <python_version>+freethreaded python3 <file.py>
```

* `<python_version>`: Replace this with the specific Python version you want to test (e.g., 3.13, 1.14, etc.). If the 
* specified version is not found on your system, `uv` will automatically download and set it up in the environment.
* `<file.py>`: This is the Python file you want to execute

## Tests

To run the unit tests and check the coverage, you can use the following command:
```bash
pytest [--pure]
```

Use `--pure` to ignore compiled runtime files and use pure Python files to provide easier coverage and test the pure 
files. Note that if no build has been performed, this parameter has no effect, as only pure runtime exist.

\* Test dependencies are required, and pip only installs project dependencies. Use `pip install --group test` to 
install them.

To manage all dependencies, it is recommended to run the tests with [uv](https://docs.astral.sh/uv/getting-started/installation/): 
```bash
uv run [-p <python_version>+freethreaded] pytest [--pure]
```

## Development Guide (uv)

Install dependencies and [re-]compile the runtime: `uv sync --reinstall-package omp4py`

Build the project and generate the wheel: `uv build`

