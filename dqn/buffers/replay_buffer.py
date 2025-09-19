import random
import numpy as np
from numpy.typing import NDArray

from ..experiences import Experience, ExperiencesBatch
from ..utils.types import IntArray

from .circular_buffer import CircularBuffer


class ReplayBuffer:
    """A class representing a basic replay buffer for storing experiences."""

    def __init__(self, max_size: int) -> None:
        """Initializes the replay buffer with a maximum size.

        Args:
            max_size: The maximum number of experiences to store in the buffer.
        """
        self.max_size = int(max_size)
        self.buffer: CircularBuffer[Experience] = CircularBuffer(max_size=self.max_size)

    @property
    def size(self) -> int:
        """The current size of the replay buffer."""
        return self.buffer.size

    def add(self, exp: Experience) -> None:
        """Add a single experience to the replay buffer.

        Args:
            exp: The experience to be added to the buffer.
        """
        self.buffer.add(exp)

    def add_batch(self, batch: ExperiencesBatch) -> None:
        """Add a batch of experiences to the replay buffer.

        Args:
            batch: A batch of experiences to be added to the buffer.
        """
        experiences = batch.to_experiences()
        for exp in experiences:
            self.buffer.add(exp)

    def get(self, index: int) -> Experience:
        """Return the experience at the given logical index."""
        return self.buffer.get(index)

    def get_batch(self, indices: IntArray) -> ExperiencesBatch:
        experiences = [self.buffer.get(i) for i in indices]
        batch = ExperiencesBatch.from_experiences(experiences, indices)
        return batch

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of sampled experiences.
        """
        experiences = self.buffer.sample(batch_size)
        batch = ExperiencesBatch.from_experiences(experiences)
        return batch
