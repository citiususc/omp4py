# OMP4py: An OpenMP Implementation for Python

**OMP4py** is a Python library that provides an implementation of OpenMP, enabling easy and efficient parallel programming for scientific and engineering applications. With omp4py, you can leverage multicore systems to improve the performance of your programs using a familiar and simple interface inspired by OpenMP.

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

def main():

    with omp("parallel num_threads(2)"):
        print(f"Hello, World from thread {omp_get_thread_num()}")

if __name__ == "__main__":
    main()