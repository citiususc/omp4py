import networkx as nx
import time
from omputils import njit, pyomp, omp, omp_pure, use_pyomp, use_pure, use_compiled, use_compiled_types

if use_pure():
    omp = omp_pure


@omp(compile=use_compiled())
def _omp4py_graphc(G):
    clustering = [0] * len(G)

    with omp("parallel for"):
        for i in range(len(clustering)):
            clustering[i] = nx.clustering(G, nodes=i)
    return clustering


def graphc(n=300000, seed=0):
    print(f"graphc: n={n}, seed={seed}")
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering
    G = nx.barabasi_albert_graph(n, min(100, n - 1))
    if use_pyomp():
        raise RuntimeError('PyOmp cannot handle networkx library')

    wtime = time.perf_counter()
    _omp4py_graphc(G)
    wtime = time.perf_counter() - wtime

    print("Elapsed time : %.6f" % wtime)
