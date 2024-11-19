# OMP4Py: a pure Python implementation of OpenMP

**OMP4Py** is a Python library that provides an implementation of OpenMP, which is widely recognized as the standard programming model for exploiting multithreading parallelism in HPC. OMP4Py brings OpenMP’s familiar directive-based parallelization paradigm to Python, allowing developers to write parallel code with the same level of control and flexibility as in C, C++, or Fortran.

## Features

- Native Python library
- Simplified parallelization of loops and code sections
- Efficient thread management and synchronization
- API compliant with OpenMP 3.0 standard

## Installation

You can install omp4py via pip:

```bash
pip install omp4py
```

## Usage

Here's a basic example of how to use omp4py:

```python
from omp4py import *

@omp
def main():

    with omp("parallel num_threads(2)"):
        print(f"Hello, World from thread {omp_get_thread_num()}")

if __name__ == "__main__":
    main()
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
