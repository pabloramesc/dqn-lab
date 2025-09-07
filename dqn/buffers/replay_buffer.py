from collections import deque

import numpy as np

from ..experiences import Experience, ExperiencesBatch


class ReplayBuffer:
    """A class representing a basic replay buffer for storing experiences."""

    def __init__(self, max_size: int) -> None:
        """Initializes the replay buffer with a maximum size.

        Args:
            max_size: The maximum number of experiences to store in the buffer.
        """
        self.max_size = int(max_size)
        self.buffer: deque[Experience] = deque(maxlen=self.max_size)

    def add(self, exp: Experience) -> None:
        """Add a single experience to the replay buffer.

        Args:
            exp: The experience to be added to the buffer.
        """
        self.buffer.append(exp)

    def add_batch(self, batch: ExperiencesBatch) -> None:
        """Add a batch of experiences to the replay buffer.

        Args:
            batch: A batch of experiences to be added to the buffer.
        """
        experiences = batch.to_experiences()
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of sampled experiences.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self._get_batch_by_indices(indices)
        return batch

    def _get_batch_by_indices(self, indices: np.ndarray) -> ExperiencesBatch:
        experiences = [self.buffer[i].to_tuple() for i in indices]
        states, actions, next_states, rewards, dones = zip(*experiences)
        batch = ExperiencesBatch(
            states=np.array(states),
            actions=np.array(actions),
            next_states=np.array(next_states),
            rewards=np.array(rewards),
            dones=np.array(dones),
            indices=indices,
            weights=None,
        )
        return batch

    @property
    def size(self) -> int:
        """The current size of the replay buffer."""
        return len(self.buffer)

    def __len__(self) -> int:
        """The current size of the replay buffer."""
        return len(self.buffer)
