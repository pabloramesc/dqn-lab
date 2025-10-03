import random
from typing import Generic, List, TypeVar

T = TypeVar("T")  # Type of elements stored in buffer


class CircularBuffer(Generic[T]):
    """Fixed-size circular buffer for storing arbitrary objects."""

    def __init__(self, max_size: int) -> None:
        """Initialize the buffer.

        Args:
            max_size: Maximum number of items the buffer can hold.
        """
        self._max_size = max_size
        self._buffer: List[T] = []
        self._index = 0

    @property
    def size(self) -> int:
        """Number of items currently stored."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its maximum capacity."""
        return len(self._buffer) == self._max_size

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._buffer.clear()
        self._index = 0

    def add(self, item: T) -> None:
        """Add an item to the buffer.

        Overwrites the oldest item when the buffer is full.

        Args:
            item: The item to add.
        """
        if self.size < self._max_size:
            self._buffer.append(item)
        else:
            self._buffer[self._index] = item
        self._index = (self._index + 1) % self._max_size

    def get(self, index: int) -> T:
        """Get an item by chronological index.

        Index `0` corresponds to the oldest element currently stored,
        while `size-1` or `-1` corresponds to the newest.
        Negative indices count from the newest backwards.

        Args:
            index: The chronological index.

        Returns:
            The item at the given chronological index.

        Raises:
            IndexError: If the index is out of range or the buffer is empty.
        """
        if self.size == 0:
            raise IndexError("Buffer is empty.")

        if index < 0:
            index += self.size

        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range.")

        if self.is_full:
            index = (self._index + index) % self._max_size

        return self._buffer[index]

    def get_physical(self, index: int) -> T:
        """Get an item by its physical slot index.

        This accesses the raw storage slot (0..max_size-1) directly,
        regardless of chronological order.

        Args:
            index: The physical slot index.

        Returns:
            The item at the given physical index.

        Raises:
            IndexError: If the index is out of range or the slot is not yet filled.
        """
        if index < 0 or index >= self._max_size:
            raise IndexError(f"Physical index {index} out of range.")

        if index >= self.size:
            raise IndexError(f"Physical index {index} not yet filled.")

        return self._buffer[index]

    def sample(self, size: int) -> List[T]:
        """Randomly sample items from the buffer with replacement.

        Args:
            size: Number of items to sample.

        Returns:
            A list of sampled items. Empty list if buffer is empty.
        """
        if self.size == 0:
            return []
        return random.choices(self._buffer, k=size)

    def to_list(self) -> List[T]:
        """Return a list of items from oldest to newest.

        Returns:
            A list containing the buffer's contents in chronological order.
        """
        if not self.is_full:
            return self._buffer
        # Full buffer: reorder oldest → newest
        return self._buffer[self._index :] + self._buffer[: self._index]
