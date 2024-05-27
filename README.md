# OMP4py: An OpenMP Implementation for Python

**OMP4py** is a Python library that provides an implementation of OpenMP, enabling easy and efficient parallel programming for scientific and engineering applications. With OMP4py, you can leverage multicore systems to improve the performance of your programs using a familiar and simple interface inspired by OpenMP.

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