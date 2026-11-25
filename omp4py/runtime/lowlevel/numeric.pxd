from cpython cimport array

# Size in bytes that separates data that different threads write, so they do not share a cache line (false sharing).
# It is fixed to 128: it covers the 64 bytes of x86 (whose prefetcher reads pairs of lines) and the 128 of Apple and arm64.
cdef enum:
    CACHE_LINE = 128

# Integer
ctypedef long long pyint
ctypedef pyint[:] pyint_array

# Floating point
ctypedef double pyfloat
ctypedef pyfloat[:] pyfloat_array


# New Integer Array
cdef array.array _pyint_array_template
cdef inline pyint_array new_pyint_array(pyint n):
    return array.clone(_pyint_array_template, n, True)

# New Floating point Array
cdef array.array _pyfloat_array_template
cdef inline pyfloat_array new_pyfloat_array(pyint n):
    return array.clone(_pyfloat_array_template, n, True)
