import os
import math
import time
import numpy as np
from omputils import njit, pyomp, omp, use_pyomp, use_compiled, use_compiled_types, get_omp_threads

try:
    import cython
except ImportError:
    pass

UP = 0
DOWN = 2
RIGHT = 1
LEFT = 3

MOVES = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]


@njit
def _pyomp_rmaze(maze, start, path, step, result):
    if result[0] > -1:
        return
    for (i, m) in enumerate(MOVES):
        other = start + m
        if 0 <= other[0] < maze.shape[0] and 0 <= other[1] < maze.shape[1] and maze[other[0], other[1]] == 0:
            if step == 0 or (i == path[step - 1] or path[step - 1] % 2 != i % 2):
                new_path = np.append(path, i)
                with pyomp("task firstprivate(i, new_path)"):
                    other = start + MOVES[i]
                    if other[0] == maze.shape[0] - 1 and other[1] == maze.shape[1] - 1:
                        result[:len(new_path)] = new_path

                    _pyomp_rmaze(maze, other, new_path, step + 1, result)


@njit
def _pyomp_maze(maze):
    result = np.ones(len(maze) * len(maze)) * -1
    with pyomp("parallel"):
        with pyomp("single"):
            path = np.ones(len(maze) * len(maze)) * -1
            _pyomp_rmaze(maze, np.array([0, 0]), path, 0, result)
    return result


@omp(compile=use_compiled())
def _omp4py_mraze(maze, start, path, step, result):
    if result[0] > -1:
        return
    for (i, m) in enumerate(MOVES):
        other = start + m
        if 0 <= other[0] < maze.shape[0] and 0 <= other[1] < maze.shape[1] and maze[other[0], other[1]] == 0:
            if step == 0 or (i == path[step - 1] or path[step - 1] % 2 != i % 2):
                new_path = np.append(path, i)
                with omp("task firstprivate(i, new_path)"):
                    other = start + MOVES[i]
                    if other[0] == maze.shape[0] - 1 and other[1] == maze.shape[1] - 1:
                        result[:len(new_path)] = new_path

                    _omp4py_mraze(maze, other, new_path, step + 1, result)


@omp(compile=use_compiled())
def _omp4py_maze(maze):
    result = np.ones(len(maze) * len(maze), dtype=np.int8) * -1
    with omp("parallel"):
        with omp("single"):
            path = np.ones(0, dtype=np.int8) * -1
            _omp4py_mraze(maze, np.array([0, 0]), path, 0, result)
    return result


@omp(compile=use_compiled())
def _omp4py_mraze_types(maze: np.ndarray, start: np.ndarray, path: np.ndarray, step: int, result: np.ndarray):
    if result[0] > -1:
        return
    i: int
    m: np.ndarray
    other: np.ndarray
    for (i, m) in enumerate(MOVES):
        other = start + m
        if 0 <= other[0] < maze.shape[0] and 0 <= other[1] < maze.shape[1] and maze[other[0], other[1]] == 0:
            if step == 0 or (i == path[step - 1] or path[step - 1] % 2 != i % 2):
                new_path = np.append(path, i)
                with omp("task firstprivate(i, new_path)"):
                    other = start + MOVES[i]
                    if other[0] == maze.shape[0] - 1 and other[1] == maze.shape[1] - 1:
                        result[:len(new_path)] = new_path

                    _omp4py_mraze_types(maze, other, new_path, step + 1, result)


@omp(compile=use_compiled())
def _omp4py_maze_types(maze):
    result = np.ones(len(maze) * len(maze), dtype=np.int8) * -1
    with omp("parallel"):
        with omp("single"):
            path = np.ones(0, dtype=np.int8) * -1
            _omp4py_mraze_types(maze, np.array([0, 0]), path, 0, result)
    return result


def _gen_rwalls(n, seed):
    np.random.seed(seed)
    rank = np.zeros((n, n), dtype=int)
    groups = list()
    edges = [((x, y), (x + 1, y)) for y in range(n) for x in range(n - 1)] + \
            [((x, y), (x, y + 1)) for y in range(n - 1) for x in range(n)]
    np.random.shuffle(edges)

    for i in range(n):
        for j in range(n):
            rank[i, j] = i * n + j
            groups.append({(i, j)})

    remove_wall = list()
    while len(edges) > 0:
        x, y = edges.pop()
        a, b = rank[x[0], x[1]], rank[y[0], y[1]]

        if a != b:
            groups[a].update(groups[b])
            for i in groups[b]:
                rank[i] = a
            groups[b] = None
            remove_wall.append((x, y))

    return remove_wall


def _gen_maze(n, seed):
    walls = int(math.ceil((n - 1) / 2))
    n = walls * 2 + 1
    remove_wall = _gen_rwalls(n - walls, seed)

    maze = np.ones((n, n), dtype=np.int8)
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            maze[i, j] = 0

    for (x0, x1), (y0, y1) in remove_wall:
        x0, x1 = 2 * x0, 2 * x1
        y0, y1 = 2 * y0, 2 * y1
        if x0 == y0:
            maze[x0][min(x1, y1) + 1] = 0
        elif x1 == y1:
            maze[min(x0, y0) + 1][x1] = 0

    return maze


def maze(n=2100, seed=0, cache="maze"):
    if cache is not None and os.path.exists(f"{cache}.npy"):
        maze = np.load(f"{cache}-{n}-{seed}.npy")
    else:
        maze = _gen_maze(n, seed)
        if cache is not None:
            np.save(f"{cache}-{n}-{seed}.npy", maze)

    print(f"maze : n={len(maze)}, seed={seed}, cache={cache}")

    #for line in maze:
    #    print("  ".join(map(str, line.tolist())))

    wtime = time.perf_counter()
    if use_pyomp():
        result = _pyomp_maze(maze)
    elif use_compiled_types():
        result = _omp4py_maze_types(maze)
    else:
        result = _omp4py_maze(maze)
    wtime = time.perf_counter() - wtime

    print(f"   Path len  : {np.where(result == -1)[0][0]}")
    print("Elapsed time : %.6f" % wtime)
