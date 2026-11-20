from omp4py import *

omp(extension="modifier_ext")


@omp(modifiers={"strip_asserts": True}, dump="stripped.py")
def check(x):
    assert x > 0, "x must be positive"
    return x


print(check(-5))


@omp(modifiers={"xrange": True})
def squares(n):
    out = [0] * n
    with omp("parallel for num_threads(2)"):
        for i in xrange(n):                      # noqa: F821  (rewritten to range before the transformer)
            out[i] = i * i
    return out


print(squares(5))
