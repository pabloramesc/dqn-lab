from typing import Union
import numpy as np

FloatLike = Union[float, np.floating]


class SumTree:
    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data_ptr = 0
        self._size = 0

    @property
    def total_priority(self) -> float:
        return self._tree[0]

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, priority: FloatLike):
        """Add a new priority to the tree with circular replacement."""
        idx = self._data_ptr + self._capacity - 1
        self._update_leaf(idx, priority)
        self._data_ptr = (self._data_ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self):
        """Sample a leaf proportional to its priority."""
        if self.total_priority == 0:
            raise ValueError("Cannot sample from an empty sum-tree.")

        r = np.random.uniform(0, self.total_priority)
        leaf_idx, priority = self._get_leaf(r)  # type: ignore
        data_idx = leaf_idx - (self._capacity - 1)
        return data_idx, priority

    def get_priority(self, data_idx: int) -> float:
        """Return the current priority value for the given data index."""
        leaf_idx = data_idx + self._capacity - 1
        return self._tree[leaf_idx]

    def update_priority(self, data_idx: int, priority: FloatLike):
        """Update a priority for a given buffer index."""
        leaf_idx = data_idx + self._capacity - 1
        self._update_leaf(leaf_idx, priority)

    def check_consistency(self, idx: int = 0) -> bool:
        """Recursively check if each node equals the sum of its children."""
        # If leaf node, return consistent
        if idx >= self._capacity - 1:
            return True

        left = 2 * idx + 1
        right = left + 1

        # Compute expected sum
        expected = self._tree[left] + self._tree[right]
        node_ok = np.isclose(self._tree[idx], expected)

        # Recursively check children
        left_ok = self.check_consistency(left)
        right_ok = self.check_consistency(right)

        return node_ok and left_ok and right_ok

    def _update_leaf(self, idx: int, priority: FloatLike):
        # Update leaf value
        self._tree[idx] = priority

        # Propagete change to root
        while idx > 0:
            idx = (idx - 1) // 2
            left = 2 * idx + 1
            right = left + 1
            self._tree[idx] = self._tree[left] + self._tree[right]

    def _get_leaf(self, value: FloatLike) -> tuple[int, float]:
        idx = 0
        while idx < self._capacity - 1:  # not a leaf
            left = 2 * idx + 1
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        return idx, self._tree[idx]
