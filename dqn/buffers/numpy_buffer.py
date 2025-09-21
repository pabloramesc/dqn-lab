from typing import Tuple, Union, Any

import numpy as np
from numpy.typing import NDArray


class NumpyBuffer:
    """
    Fixed-size circular buffer for storing NumPy arrays of arbitrary shape and dtype.
    """

    def __init__(
        self, max_size: int, shape: Union[int, Tuple[int, ...]] = (), dtype=np.float32
    ) -> None:
        """
        Initialize a NumpyBuffer.

        Args:
            max_size: Maximum number of elements in the buffer.
            shape: Shape of a single element (excluding batch dimension).
            dtype: NumPy data type of the buffer elements.
        """
        self._max_size = int(max_size)
        self._shape = shape if isinstance(shape, tuple) else (shape,)
        self._dtype = dtype

        self._buffer = np.zeros((self._max_size, *self._shape), dtype=self._dtype)
        self._index = 0
        self._size = 0

    @property
    def max_size(self) -> int:
        """Maximum capacity of the buffer."""
        return self._max_size

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of a single element in the buffer."""
        return self._shape

    @property
    def dtype(self) -> type:
        """Data type of the buffer elements."""
        return self._dtype

    @property
    def size(self) -> int:
        """Current number of elements stored in the buffer."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        return self._size == self._max_size

    def add(self, item: Any) -> None:
        """Add an element to the buffer.

        Args:
            item: The element to add. Must match the buffer shape and dtype.

        Raises:
            ValueError: If the shape of the element does not match the buffer shape.
        """
        item = self._process_item(item)
        self._buffer[self._index] = item
        self._index = (self._index + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def get(self, index: int) -> NDArray:
        """Get an element from the buffer.

        Args:
            index: Logical index of the element. Supports negative indexing.

        Returns:
            The element at the given index.

        Raises:
            IndexError: If the buffer is empty or index is out of range.
        """
        index = self._resolve_index(index)
        return self._buffer[index]

    def set(self, index: int, item: Any) -> None:
        """Set an element at a given index.

        Args:
            index: Logical index of the element. Supports negative indexing.
            item: New value to store. Must match buffer shape and dtype.

        Raises:
            ValueError: If the shape of the element does not match the buffer shape.
            IndexError: If the index is out of range.
        """
        index = self._resolve_index(index)
        item = self._process_item(item)
        if item.shape != self._shape:
            raise ValueError(f"Expected shape {self._shape}, got {item.shape}")
        self._buffer[index] = item

    def to_array(self) -> NDArray:
        """Return all buffer contents as a contiguous array from oldest to newest.

        Returns:
            NumPy array of all elements in the buffer.
        """
        if not self.is_full:
            return self._buffer[: self._size].copy()
        return np.concatenate(
            (self._buffer[self._index :], self._buffer[: self._index]), axis=0
        )

    def _resolve_index(self, index: int) -> int:
        if self._size == 0:
            raise IndexError("Buffer is empty.")

        if index < 0:
            index += self._size

        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range.")

        if self.is_full:
            index = (self._index + index) % self._max_size

        return index

    def _process_item(self, item: Any) -> NDArray:
        item = np.asarray(item, dtype=self._dtype)
        if item.shape != self._shape:
            raise ValueError(f"Expected shape {self._shape}, got {item.shape}")
        return item

    def __getitem__(self, index: int) -> NDArray:
        return self.get(index)

    def __setitem__(self, index: int, value: NDArray) -> None:
        self.set(index, value)
