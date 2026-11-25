"""`workqueue` package: structures that distribute the work of the tasks among the threads of a team.

This package holds the pieces on which the tasking runtime is built. They are written with the primitives of the
`lowlevel` package (atomics, mutexes, events) and have no C code of their own: the same source is interpreted in
pure Python mode and compiled with Cython, using the interface of its `.pxd` file.

- `queue`: the queue of ready tasks of a thread, which the owner uses as a stack and the other threads steal from.
- `rng`: per-thread random numbers, to pick the thread to steal from.
- `sleeper`: threads that have no work and sleep until somebody wakes them up.
"""
