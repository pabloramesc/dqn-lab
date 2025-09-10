from typing import Union, SupportsFloat
import numpy as np

FloatLike = Union[float, np.floating]


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_ptr = 0

    def add(self, priority: FloatLike):
        """Add a new priority to the tree with circular replacement."""
        idx = self.data_ptr + self.capacity - 1
        self._update_leaf(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity

    def sample(self):
        """Sample a leaf proportional to its priority."""
        if self.total_priority == 0:
            raise ValueError("Cannot sample from an empty sum-tree.")

        r = np.random.uniform(0, self.total_priority)
        leaf_idx, priority = self._get_leaf(r)
        data_idx = leaf_idx - (self.capacity - 1)
        return data_idx, priority

    def update_priority(self, data_idx: int, priority: FloatLike):
        """Update a priority for a given buffer index."""
        leaf_idx = data_idx + self.capacity - 1
        self._update_leaf(leaf_idx, priority)

    def _update_leaf(self, leaf_idx: int, priority: FloatLike):
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        # propagate change to root
        while leaf_idx != 0:
            leaf_idx = (leaf_idx - 1) // 2
            self.tree[leaf_idx] += change

    def _get_leaf(self, value: FloatLike):
        leaf_idx = 0
        while leaf_idx < self.capacity - 1:  # not a leaf
            left = 2 * leaf_idx + 1
            right = left + 1
            if value <= self.tree[left]:
                leaf_idx = left
            else:
                value -= self.tree[left]
                leaf_idx = right
        return leaf_idx, self.tree[leaf_idx]

    @property
    def total_priority(self):
        return self.tree[0]
