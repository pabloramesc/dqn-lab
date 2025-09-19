import random
from typing import Generic, List, TypeVar

T = TypeVar("T")  # Type of elements stored in buffer


class CircularBuffer(Generic[T]):
    """Fixed-size circular buffer for storing arbitrary objects."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._buffer: List[T] = []
        self._index = 0

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._max_size

    def clear(self) -> None:
        self._buffer.clear()
        self._index = 0

    def add(self, item: T) -> None:
        if self.size < self._max_size:
            self._buffer.append(item)
        else:
            self._buffer[self._index] = item
        self._index = (self._index + 1) % self._max_size

    def get(self, index: int) -> T:
        if self.size == 0:
            raise IndexError("Buffer is empty.")

        if index < 0:
            index += self.size

        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range.")

        if self.is_full:
            index = (self._index + index) % self._max_size

        return self._buffer[index]

    def sample(self, size: int) -> List[T]:
        if self.size == 0:
            return []
        return random.choices(self._buffer, k=size)

    def to_list(self) -> List[T]:
        if not self.is_full:
            return self._buffer
        # Full buffer: reorder oldest → newest
        return self._buffer[self._index :] + self._buffer[: self._index]
