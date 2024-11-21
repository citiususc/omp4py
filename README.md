# OMP4Py: a pure Python implementation of OpenMP

**OMP4Py** is a Python library that provides an implementation of OpenMP, which is widely recognized as the standard programming model for exploiting multithreading parallelism in HPC. OMP4Py brings OpenMP’s familiar directive-based parallelization paradigm to Python, allowing developers to write parallel code with the same level of control and flexibility as in C, C++, or Fortran.

## Features

- Native Python library
- Simplified parallelization of loops and code sections
- Efficient thread management and synchronization
- API compliant with [OpenMP 3.0 standard](https://www.openmp.org/wp-content/uploads/spec30.pdf)

## Installation

You can install omp4py via pip:

```bash
pip install omp4py
```
**Note**: OMP4Py is compatible with Python versions that include the Global Interpreter Lock (GIL). However, to fully exploit multithreading for scaling applications, it is necessary to use Python 3.13 or later, which offers a no-GIL option.

## Usage

OMP4Py defines a function `omp` that operates similarly to OpenMP directives in C/C++, maintaining the same syntax and functionality. The function itself has no effect when executed; it serves solely as a container for the OpenMP directives. Note that when a OpenMP directive must be used within structured blocks, the `omp` function is used together as part of a `with` block; otherwise, it is used as a standalone function call. Note that we must
decorate a function or class containing the OpenMP directives with the `@omp` decorator.

Here's a basic example of how to use OMP4Py to calculate $\pi$:

```python
from omp4py import *
import random 
    
@omp
def pi(num_points):
    count = 0
    with omp("parallel for reduction(+:count)"):
        for i in range(num_points):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1.0:
                count += 1
    pi = count / num_points
    return pi

print(pi(10000000))  
```

## Tests

To run the unit tests and check the coverage, you can use the following commands with Poetry:

1. **Run the unit tests:**

    ```bash
     poetry run coverage run
    ```

2. **Generate a coverage report:**

    ```bash
     poetry run coverage html
    ```
