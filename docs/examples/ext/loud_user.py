from omp4py import *

omp(extension="loud_device")


@omp(device="loud")
def run():
    with omp("parallel num_threads(2)"):
        with omp("single"):
            print("only one thread prints this")
        omp("barrier")


run()
