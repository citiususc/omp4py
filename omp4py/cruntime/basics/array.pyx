from cpython.array cimport array, resize, copy as copy2

cdef array[pyint] new_int(pyint n):
    a: array[pyint] = array('q')
    resize(a, n)
    return a

cdef array[double] new_float(pyint n):
    a: array[pyint] = array('d')
    resize(a, n)
    return a

cdef array[pyint] int_from(list[pyint] elems):
    return array('q', elems)

cdef array[double] float_from(list[double] elems):
    return array('d', elems)

cdef iview as_iview(elems: array[pyint]):
    return elems

cdef fview as_fview(elems: array[double]):
    return elems

cdef array copy(array elems):
    return copy2(elems)
