import numpy as np
import random
from numpy.typing import NDArray

from .replay_buffer import ReplayBuffer
from ..experiences import Experience, ExperiencesBatch
from ..types import IntArray


class CircularBuffer(ReplayBuffer):
    """An efficient circular buffer for storing and sampling experiences."""

    def __init__(self, max_size: int):
        self._max_size = int(max_size)
        self._buffer: NDArray = np.empty(self._max_size, dtype=object)  # type: ignore
        # self._buffer: list[Experience] = [None] * self._max_size  # type: ignore
        self._size = 0
        self._index = 0

    def add(self, exp: Experience):
        self._buffer[self._index] = exp
        self._size = min(self._size + 1, self._max_size)
        self._index = (self._index + 1) % self._max_size

    def add_batch(self, batch: ExperiencesBatch):
        experiences = batch.to_experiences()
        for exp in experiences:
            self.add(exp)

    def get(self, index: int) -> Experience:
        if self._size == 0:
            raise IndexError("Buffer is empty.")

        if index < 0:
            index += self._size

        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range.")

        if self._size == self._max_size:
            index = (index + self._index) % self._max_size

        return self._buffer[index]

    def get_batch(self, indices: IntArray) -> ExperiencesBatch:
        # experiences = [self._buffer[i] for i in indices]
        indices = np.asarray(indices, dtype=np.int32)
        experiences = self._buffer[indices]
        batch = ExperiencesBatch.from_experiences(experiences, indices)  # type: ignore
        return batch

    def sample(self, batch_size: int) -> ExperiencesBatch:
        experiences = random.choices(self._buffer[: self._size], k=batch_size)
        batch = ExperiencesBatch.from_experiences(experiences)
        return batch

    @property
    def size(self) -> int:
        return self._size
