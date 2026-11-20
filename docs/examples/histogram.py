from collections import Counter

from omp4py import *


@omp
def histogram_critical(words):
    """Simple but slow: every update takes the lock."""
    counts = Counter()

    with omp("parallel for schedule(dynamic, 64) num_threads(4)"):
        for i in range(len(words)):
            with omp("critical"):
                counts[words[i]] += 1

    return counts


@omp
def histogram_partial(words):
    """Faster: each thread counts privately, then merges once."""
    counts = Counter()

    with omp("parallel num_threads(4)"):
        local = Counter()                       # created inside the region: one per thread
        with omp("for schedule(static) nowait"):
            for i in range(len(words)):
                local[words[i]] += 1
        with omp("critical"):
            counts.update(local)                # one short critical section per thread

    return counts


words = ("alpha beta gamma beta alpha beta " * 500).split()
print(sorted(histogram_critical(words).items()))
print(sorted(histogram_partial(words).items()))
