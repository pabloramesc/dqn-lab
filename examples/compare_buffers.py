import timeit

import numpy as np

from dqn.buffers import (
    BufferProtocol,
    CircularBuffer,
    OptimizedPER,
    PERBuffer,
    ReplayBuffer,
)
from dqn.experiences import Experience, ExperiencesBatch

BUFFER_SIZE = 1_000_000
BATCH_SIZE = 64
STATE_SHAPE = (10,)
TIMEIT_RUNS = 1000


def benchmark_add(buff: BufferProtocol, exp: Experience):
    """Callable wrapper for adding a experience."""

    def func():
        buff.add(exp)

    return func


def benchmark_get(buff: BufferProtocol, idx: int):
    """Callable wrapper for getting a experience."""

    def func():
        _ = buff.get(idx)

    return func


def benchmark_add_batch(buff: BufferProtocol, batch: ExperiencesBatch):
    """Callable wrapper for adding a batch of experiences."""

    def func():
        buff.add_batch(batch)

    return func


def benchmark_sample(buff: BufferProtocol, batch_size: int):
    """Callable wrapper for sampling random experiences."""

    def func():
        _ = buff.sample(batch_size)

    return func


print("Creating dummy experiences...")
dummy_experiences = [
    Experience(
        state=np.full(shape=STATE_SHAPE, fill_value=float(i)),
        action=int(i),
        next_state=np.zeros(STATE_SHAPE),
        reward=float(i),
        done=False,
    )
    for i in range(BUFFER_SIZE)
]
dummy_batch = ExperiencesBatch.from_experiences(dummy_experiences)

buffers: dict[str, BufferProtocol] = {
    "deque": ReplayBuffer(max_size=BUFFER_SIZE),
    "circular": CircularBuffer(max_size=BUFFER_SIZE),
    "per": PERBuffer(max_size=BUFFER_SIZE),
    "optimized-per": OptimizedPER(max_size=BUFFER_SIZE),
}

print("Filling buffers...")
for buff in buffers.values():
    buff.add_batch(dummy_batch)

# Run benchmarks
print("\n=== Add experience timing ===")
for name, buff in buffers.items():
    exp = dummy_experiences[0]
    t = timeit.timeit(benchmark_add(buff, exp), number=TIMEIT_RUNS)
    print(f"{name}: {t*1e3:.4f} ms")

print("\n=== Get experience timing ===")
for name, buff in buffers.items():
    idx = 0
    t = timeit.timeit(benchmark_get(buff, idx), number=TIMEIT_RUNS)
    print(f"{name}: {t*1e3:.4f} ms")

print("\n=== Add batch timing ===")
for name, buff in buffers.items():
    batch = ExperiencesBatch.from_experiences(dummy_experiences[:BATCH_SIZE])
    t = timeit.timeit(benchmark_add_batch(buff, batch), number=TIMEIT_RUNS)
    print(f"{name:10}: {t:.4f} s")

print("\n=== Sample batch timing ===")
for name, buff in buffers.items():
    t = timeit.timeit(benchmark_sample(buff, BATCH_SIZE), number=TIMEIT_RUNS)
    print(f"{name:10}: {t:.4f} s")
