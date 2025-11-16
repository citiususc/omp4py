# **OMP4Py Examples**

Install dependencies (first time): ``pip install poetry`` and ``poetry install``

Use Poetry environment: ``poetry env activate``

Run: ``python3 main.py  <mode> <test> <threads> [args...]``

Available modes: 0 -> pure, 1 -> hybrid, 2 -> compiled, 3 -> compiled with data types

Available tests: fft, graphc, fib, jacobi, lud, maze, md, pi, qsort, quad, wordcount

### 1. **Fast Fourier Transform**
This experiment evaluates the performance of the Fast Fourier Transform (FFT), an efficient algorithm used to compute the Discrete Fourier Transform (DFT) of a sequence. FFT allows for the conversion of a signal from the time domain to the frequency domain. The tests were conducted using a complex data vector of 4 million numbers.

``python3 main.py 1 fft 2 [ln2_max=22] [nits=10000] [seed=331]``


### 2. **Graph Clustering**
The clustering coefficient of a node in an unweighted graph is the fraction of possible triangles passing through that node. The experiment used a graph with 300k vertices, each connected by 100 edges. The graph generation, storage, and clustering algorithm were implemented using the NetworkX library. It is important to note that PyOMP cannot run this benchmark because Numba is unable to compile the `Graph` object and the clustering algorithm function calls, as they are part of an external library not optimized for Numba.

``python3 main.py 1 graphc 2 [n=300000] [seed=0]``


### 2. **Fibonacci Sequence**

This experiment evaluates the performance of algorithms used to compute the **Fibonacci sequence**, a series of numbers where each term is the sum of the two preceding ones. The sequence is fundamental in mathematics and computer science, appearing in various applications such as algorithm analysis, dynamic programming, and modeling natural phenomena. The sequence is defined recursively as:
`F(0) = 0 ` `F(1) = 1 ` `F(n) = F(n−1) + F(n−2), for n > 1`

``python3 main.py 1 fib 2 [n=30]``


### 3. **Jacobi Method**
The Jacobi method is an iterative algorithm used for solving systems of linear equations of the form `A · x = b`, where `A` is a matrix, and `x` and `b` are vectors. At each iteration, the solution vector is updated based on values from the previous iteration. For this experiment, a square matrix `A` of size `1k × 1k` was used, performing up to 1,000 iterations with a stopping criterion set to an error tolerance of `1 × 10^(-6)`.

``python3 main.py 1 jacobi 2 [n=1000] [max_iter=1000] [tol=1e-6] [seed=0]``

MPI: ``poetry install mpi4py`` and use ``mpirun -n <procs> python3 main.py 1 jacobi 2 [args...]``.


### 4. **LU Decomposition**
LU decomposition is a method for factorizing a matrix `A` into the product of a lower triangular matrix `L` and an upper triangular matrix `U`, such that `A = L · U`. This factorization simplifies solving systems of linear equations, matrix inversion, and determinant computation. For the experiment, LU decomposition was applied to a square matrix of size `1k × 1k`.

``python3 main.py 1 lud 2 [n=1000] [seed=0]``


### 5. **Pathfinding (bfs)**

Solved via breadth-first search on a grid (entrance at the top-left, exit at the bottom-right). Zeros are paths and ones are walls; moves are allowed only between 0-cells, and each feasible move spawns a task

``python3 main.py 1 maze 2 [n=10] [seed=0] [file_cache=None]``


### 6. **Molecular Dynamics Simulation**
This experiment simulates the motion of particles over time using the velocity Verlet integration scheme to update positions, velocities, and accelerations. A system of 2,000 particles was simulated, interacting with a central pair potential.

``python3 main.py 1 md 2 [n=2000] [steps=10] [seed=0]``


### 7. **Riemann integration (computing π)**
The area under the curve `y = 4 / (1 + x^2)` between 0 and 1 approximates the value of `π`. This integral was estimated using numerical summation, with 2 billion intervals employed to compute the approximation.

``python3 main.py 1 pi 2 [n=2000000000]``


### 8. **Quicksort**
Quicksort is a fast, recursive **divide-and-conquer** sorting algorithm that partitions data around a pivot element.  
It recursively sorts smaller subarrays (limit argument) to achieve an **average time complexity of O(n log n)**.

``python3 main.py 1 qsort 2 [n=40000000] [limit=100000]``


### 9. **QUAD**
The QUAD experiment uses a numerical integration technique to estimate the value of an integral using an averaging method. The function `f(x) = 50 / (π * (2500 * x^2 + 1))` was approximated over the interval from `A = 0` to `B = 10`. The method involves sampling the function at numerous points within the interval to compute an average value, which is then used to estimate the integral. The test was run for 1 billion iterations.

``python3 main.py 1 quad 2 [n=1000000000]``


### 10. **Wordcount**
This algorithm counts the number of occurrences of each word in an input text. A text of 1 million characters was generated, containing words of lengths between 3 and 10 letters, with a 10% chance of a new line being added after each word. Although recent versions of Numba have experimental support for Python dictionaries, PyOMP is based on an earlier version that lacks the necessary support to compile Wordcount dictionaries.

``python3 main.py 1 wordcount 2 [n=1000000] [seed=0]`` or ``python3 main.py 1 wordcount 2 \"file.txt\"``
