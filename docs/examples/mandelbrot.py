from omp4py import *


def escape_time(c, max_iter):
    z = 0j
    n = 0
    while n < max_iter and abs(z) <= 2.0:
        z = z * z + c
        n += 1
    return n


@omp
def mandelbrot(width, height, max_iter=100):
    image = [[0] * width for _ in range(height)]

    with omp("parallel for schedule(dynamic) num_threads(4)"):
        for y in range(height):
            for x in range(width):
                c = complex(-2.0 + 3.0 * x / width, -1.0 + 2.0 * y / height)
                image[y][x] = escape_time(c, max_iter)

    return image


def mandelbrot_sequential(width, height, max_iter=100):
    return [
        [escape_time(complex(-2.0 + 3.0 * x / width, -1.0 + 2.0 * y / height), max_iter) for x in range(width)]
        for y in range(height)
    ]


if __name__ == "__main__":
    image = mandelbrot(64, 24)
    print(image == mandelbrot_sequential(64, 24))

    shades = " .:-=+*#%@"
    for row in image:
        print("".join(shades[min(n * len(shades) // 100, len(shades) - 1)] for n in row))
