cimport libc.math as math
cimport libc.stdlib as stdlib
from omp4py.cruntime.basics.types cimport *

cdef inline pyint abs(pyint n):
    return stdlib.llabs(n)

cdef inline double fabs(double n):
    return math.fabs(n)

cdef inline pyint ceil(double n):
    return <pyint> math.ceil(n)

cdef inline pyint floor(double n):
    return <pyint> math.floor(n)

cdef inline (pyint, pyint) divmod(pyint x, pyint y):
    result = stdlib.lldiv(x, y)
    return <pyint> result.quot, <pyint> result.rem
