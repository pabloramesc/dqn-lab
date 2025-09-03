import numpy as np

from ..experiences import Experience, ExperiencesBatch
from .replay_buffer import ReplayBuffer


class PriorityReplayBuffer(ReplayBuffer):
    """A class representing a prioritized replay buffer for storing experiences
    with TD errors priority based sampling.
    """

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 1e-6,
        min_priority: float = 1e-6,
    ):
        """Initializes the priority replay buffer.

        Args:
            max_size: The maximum number of experiences to store in the buffer.
            alpha: The degree of prioritization.
            beta: The degree to which importance sampling is corrected.
            beta_annealing: The rate at which beta increases over time.
            min_priority: The minimum priority value for experiences.
        """
        super().__init__(max_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_annealing = float(beta_annealing)
        self.min_priority = float(min_priority)

        # Initialize priorities circular buffer
        self.ptr = int(0)
        self.priorities = np.zeros(max_size, dtype=np.float32)

    def add(self, exp: Experience, td_error: float = 1.0) -> None:
        """Add a single experience to the replay buffer.

        Args:
            exp: The experience to be added to the buffer.
            td_error: The temporal difference error for the experience.
        """
        super().add(exp)
        priority = max(self.min_priority, td_error)
        self.priorities[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.max_size

    def add_batch(self, batch: ExperiencesBatch, td_errors: np.ndarray = None) -> None:
        """Add a batch of experiences to the replay buffer.

        Args:
            batch: A batch of experiences to be added to the buffer.
            td_errors: The temporal difference errors for each experience.
        """
        super().add_batch(batch)

        batch_size = len(batch)
        if td_errors is None:
            priority = np.ones(batch_size, dtype=np.float32)
        else:
            priority = np.clip(td_errors, a_min=self.min_priority, a_max=None)

        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.priorities[indices] = priority
        self.ptr = (self.ptr + batch_size) % self.max_size

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """Sample a batch of experiences from the replay buffer with priority sampling.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A batch of sampled experiences.
        """
        priorities = self.priorities[: self.size] ** self.alpha + self.min_priority
        probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        batch = self._get_batch_by_indices(indices)

        weights = (self.size * probabilities[indices]) ** -self.beta
        batch.weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_annealing)

        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update the priorities of experiences in the buffer based on new TD errors.

        Args:
            indices: The indices of the experiences whose priorities will be updated.
            td_errors: The new temporal difference errors used to update the priorities.
        """
        self.priorities[indices] = np.clip(
            td_errors, a_min=self.min_priority, a_max=None
        )
