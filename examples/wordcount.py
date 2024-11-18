import random
import time
import string
from omputils import njit, pyomp, omp


def random_word(min_length=3, max_length=10):
    length = random.randint(min_length, max_length)
    word = ''.join(random.choices(string.ascii_lowercase, k=length))
    return word


def generate_text(b_size, seed):
    text = ''

    random.seed(seed)
    while len(text) < b_size:
        if random.random() < 0.9:
            text += random_word()
        else:  # 10% de probabilidad de agregar un salto de línea
            text += '\n'

    return text


@njit
def _pyomp_wordcount(text):
    count = {}

    with pyomp("parallel"):
        local_count = {}
        with pyomp("for"):
            for i in range(len(text)):
                for word in text.split():
                    if word in local_count:
                        local_count[word] += 1
                    else:
                        local_count[word] = 1

        with pyomp("critical"):
            count.update(local_count)

    return count


@omp
def _omp4py_wordcount(lines):
    count = {}

    with omp("parallel"):
        local_count = {}
        with omp("for"):
            for i in range(len(lines)):
                for word in lines.split():
                    if word in local_count:
                        local_count[word] += 1
                    else:
                        local_count[word] = 1

        with omp("critical"):
            count.update(local_count)

    return count


def wordcount(b_size=1e11, seed=0, numba=False):
    print(f"wordcount: {b_size}, seed: {seed}, numba: {numba}")
    text = generate_text(b_size, seed)
    lines = text.splitlines()

    wtime = time.perf_counter()
    _pyomp_wordcount(lines) if numba else _omp4py_wordcount(lines)
    wtime = time.perf_counter() - wtime
    print("Elapsed time : %.6f" % wtime)
