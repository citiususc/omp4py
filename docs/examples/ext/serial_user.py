from omp4py import omp

omp(extension="serial_device")


@omp(device="serial")
def total(n):
    s = 0
    with omp("parallel for reduction(+:s)"):
        for i in range(n):
            s += i
    return s


print(total(100))
