import cython

# Integer
ctypedef long long pyint
ctypedef pyint[:] pyint_array

# Floating point
ctypedef double pyfloat
ctypedef pyfloat[:] pyfloat_array

# Complex
ctypedef double complex pycomplex


# New Integer Array
cdef inline pyint_array new_pyint_array(pyint n):
    return cython.view(shape=(n,), itemsize=sizeof(pyint), format="q")

# New Floating point Array
cdef inline pyfloat_array new_pyfloat_array(pyint n):
    return cython.view(shape=(n,), itemsize=sizeof(pyfloat), format="d")
