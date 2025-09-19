import random
import timeit
from abc import ABC, abstractmethod
from collections import deque
from typing import Generic, List, TypeVar

import numpy as np

T = TypeVar("T")


class BaseRingBuffer(ABC, Generic[T]):
    """Abstract base class for a ring/circular buffer."""

    @property
    def size(self) -> int: ...

    @abstractmethod
    def add(self, item: T) -> None: ...

    @abstractmethod
    def get(self, index: int) -> T: ...

    @abstractmethod
    def sample(self, size: int) -> List[T]: ...


class RingBufferNDArray(BaseRingBuffer[T]):
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._buffer = np.empty(max_size, dtype=object)
        self._size = 0
        self._index = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, item: T):
        self._buffer[self._index] = item
        self._index = (self._index + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def get(self, index: int) -> T:
        if index < 0:
            index += self._size
        if self._size == self._max_size:
            index = (index + self._index) % self._max_size
        return self._buffer[index]

    def sample(self, size: int) -> List[T]:
        indices = np.random.choice(self._size, size)
        batch = self._buffer[indices]
        return batch.tolist()


class RingBufferList(BaseRingBuffer[T]):
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._buffer: List[T] = [None] * max_size  # type: ignore
        self._size = 0
        self._index = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, item: T):
        self._buffer[self._index] = item
        self._index = (self._index + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def get(self, index: int) -> T:
        if index < 0:
            index += self._size
        if self._size == self._max_size:
            index = (index + self._index) % self._max_size
        return self._buffer[index]

    def sample(self, size: int) -> List[T]:
        return random.choices(self._buffer, k=size)


class RingBufferDeque(BaseRingBuffer[T]):
    def __init__(self, max_size: int):
        self._buffer: deque[T] = deque(maxlen=max_size)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def add(self, item: T):
        self._buffer.append(item)
        
    def get(self, index: int) -> T:
        return self._buffer[index]

    def sample(self, size: int) -> List[T]:
        return random.choices(self._buffer, k=size)


BUFFER_SIZE = 1_000_000
BATCH_SMALL = 32
BATCH_MEDIUM = 100
BATCH_LARGE = 10000


def benchmark(buffer_class: type, name: str):
    print(f"=== Benchmark {name} ===")
    buffer: BaseRingBuffer = buffer_class(BUFFER_SIZE)

    # Benchmark add()
    items = [{"value":i} for i in range(BUFFER_SIZE * 2)]
    t_add = timeit.timeit(lambda: [buffer.add(i) for i in items], number=1)
    print(f"Add {len(items)} items: {t_add:.4f} s")
    
    # Random get
    indices = np.random.randint(0, buffer.size, BUFFER_SIZE)
    t_get = timeit.timeit(lambda: [buffer.get(i) for i in indices], number=1)
    print(f"Get {len(indices)} items: {t_get:.4f} s")

    # Benchmark small batch sample
    t_small = timeit.timeit(lambda: buffer.sample(BATCH_SMALL), number=1000)
    print(f"Sample small batch ({BATCH_SMALL} items): {t_small:.4f} s")
    
    # Benchmark medium batch sample
    t_small = timeit.timeit(lambda: buffer.sample(BATCH_MEDIUM), number=1000)
    print(f"Sample medium batch ({BATCH_MEDIUM} items): {t_small:.4f} s")

    # Benchmark large batch sample
    t_large = timeit.timeit(lambda: buffer.sample(BATCH_LARGE), number=100)
    print(f"Sample large batch ({BATCH_LARGE} items): {t_large:.4f} s")

    print()


if __name__ == "__main__":
    benchmark(RingBufferNDArray, "NDArray")
    benchmark(RingBufferList, "List")
    benchmark(RingBufferDeque, "Deque")
