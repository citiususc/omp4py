from cpython cimport array

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
