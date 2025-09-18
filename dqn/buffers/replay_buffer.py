from collections import deque
import random
import numpy as np

from ..experiences import Experience, ExperiencesBatch
from ..utils.types import IntArray

class ReplayBuffer:
    """A class representing a basic replay buffer for storing experiences."""

    def __init__(self, max_size: int) -> None:
        """Initializes the replay buffer with a maximum size.

        Args:
            max_size: The maximum number of experiences to store in the buffer.
        """
        self._max_size = int(max_size)
        self._buffer: deque[Experience] = deque(maxlen=self._max_size)

    def add(self, exp: Experience) -> None:
        """Add a single experience to the replay buffer.

        Args:
            exp: The experience to be added to the buffer.
        """
        self._buffer.append(exp)

    def add_batch(self, batch: ExperiencesBatch) -> None:
        """Add a batch of experiences to the replay buffer.

        Args:
            batch: A batch of experiences to be added to the buffer.
        """
        experiences = batch.to_experiences()
        self._buffer.extend(experiences)

    def get(self, index: int) -> Experience:
        """Return the experience at the given logical index."""
        return self._buffer[index]

    def get_batch(self, indices: IntArray) -> ExperiencesBatch:
        experiences = [self._buffer[i] for i in indices]
        batch = ExperiencesBatch.from_experiences(experiences, indices)
        return batch

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of sampled experiences.
        """
        experiences = random.sample(self._buffer, k=batch_size)
        batch = ExperiencesBatch.from_experiences(experiences)
        return batch

    @property
    def size(self) -> int:
        """The current size of the replay buffer."""
        return len(self._buffer)

    def __len__(self) -> int:
        """The current size of the replay buffer."""
        return self.size
