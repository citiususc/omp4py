from omp4py import *


@omp
def collect(n):
    # classic form: the combiner is INSIDE the parentheses, after the type
    omp("declare reduction(merge : list : omp_out += omp_in) initializer(omp_priv = [])")

    items: list = []                        # the annotation selects the reduction declared for `list`
    with omp("parallel for reduction(merge:items) num_threads(4)"):
        for i in range(n):
            items.append(i)
    return sorted(items)


@omp
def longest(words):
    # clause form: the combiner is OUTSIDE the parentheses, as a clause
    omp("declare reduction(longest : object) "
        "combiner(omp_out = omp_in if len(omp_in) > len(omp_out) else omp_out) "
        "initializer(omp_priv = '')")

    best = ""
    with omp("parallel for reduction(longest:best) num_threads(3)"):
        for i in range(len(words)):
            if len(words[i]) > len(best):
                best = words[i]
    return best


print(collect(8))
print(longest(["a", "abc", "ab", "abcd", "x"]))
