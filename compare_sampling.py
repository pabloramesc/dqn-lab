import random
import timeit
import numpy as np

BUFFER_SIZE = 1_000_000
RUNS_PER_ITEM = 1_000_000

# Define small and large batch sizes
BATCH_SIZES = [1, 32, 64, 128, 1024, 100_000]

# Prepare buffers
buffer_list = [k for k in range(BUFFER_SIZE)]
buffer_array = np.array(buffer_list, dtype=object)


def sample_list_with_sample(batch_size):
    return random.sample(buffer_list, k=batch_size)


def sample_list_with_choices(batch_size):
    return random.choices(buffer_list, k=batch_size)


def sample_list_with_indices(batch_size):
    indices = np.random.choice(len(buffer_list), batch_size)
    return [buffer_list[i] for i in indices]


def sample_array_with_choices(batch_size):
    return random.choices(buffer_array, k=batch_size)


def sample_array_with_indices(batch_size):
    indices = np.random.choice(len(buffer_array), batch_size)
    return buffer_array[indices].tolist()


methods = {
    "list + random.sample": sample_list_with_sample,
    "list + random.choices": sample_list_with_choices,
    "list + random indices": sample_list_with_indices,
    "array + random.choices": sample_array_with_choices,
    "array + random indices": sample_array_with_indices,
}

for batch_size in BATCH_SIZES:
    num_runs = int(RUNS_PER_ITEM / batch_size)
    print(
        f"\n=== Benchmark: batch of {batch_size} elements run {num_runs} times ==="
    )
    for name, func in methods.items():
        t = timeit.timeit(lambda: func(batch_size), number=num_runs)
        print(f"{name}: {t:.4f} s")
