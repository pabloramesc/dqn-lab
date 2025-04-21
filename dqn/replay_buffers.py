"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from collections import deque

import numpy as np

from .experiences import Experience, ExperiencesBatch


class ReplayBuffer:
    """
    A class representing a basic replay buffer for storing experiences.
    """

    def __init__(self, max_size: int) -> None:
        """
        Initializes the replay buffer with a maximum size.

        Parameters
        ----------
        max_size : int
            The maximum number of experiences to store in the buffer.
        """
        self.max_size = int(max_size)
        self.buffer: deque[Experience] = deque(maxlen=self.max_size)

    def add(self, exp: Experience) -> None:
        """
        Add a single experience to the replay buffer.

        Parameters
        ----------
        exp : Experience
            The experience to be added to the buffer.
        """
        self.buffer.append(exp)

    def add_batch(self, batch: list[Experience]) -> None:
        """
        Add a batch of experiences to the replay buffer.

        Parameters
        ----------
        batch : list[Experience]
            A list of experiences to be added to the buffer.
        """
        self.buffer.extend(batch)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        ExperiencesBatch
            A batch of sampled experiences.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self._get_batch_by_indices(indices)
        return batch

    def _get_batch_by_indices(self, indices: np.ndarray) -> ExperiencesBatch:
        """
        Retrieve a batch of experiences given a list of indices.
        """
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
        """
        The current size of the replay buffer.
        """
        return len(self.buffer)

    def __len__(self) -> int:
        """
        The current size of the replay buffer.
        """
        return len(self.buffer)


class PriorityReplayBuffer(ReplayBuffer):
    """
    A class representing a prioritized replay buffer for storing experiences
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
        """
        Initializes the priority replay buffer with a maximum size and
        priority parameters.

        Parameters
        ----------
        max_size : int
            The maximum number of experiences to store in the buffer.
        alpha : float, optional
            The degree of prioritization (default is 0.6).
        beta : float, optional
            The degree to which importance sampling is corrected
            (default is 0.4).
        beta_annealing : float, optional
            The rate at which beta increases over time (default is 1e-6).
        min_priority : float, optional
            The minimum priority value for experiences (default is 1e-6).
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.min_priority = min_priority

        # Initialize priorities circular buffer
        self.ptr = int(0)
        self.priorities = np.zeros(max_size, dtype=np.float32)

    def add(self, exp: Experience, td_error: float = 1.0) -> None:
        """
        Add a single experience to the priority replay buffer with a priority
        based on the TD error.

        Parameters
        ----------
        exp : Experience
            The experience to be added to the buffer.
        td_error : float, optional
            The temporal difference error for the experience (default is 1.0).
        """
        super().add(exp)
        priority = max(self.min_priority, td_error)
        self.priorities[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.max_size

    def add_batch(self, batch: ExperiencesBatch, td_errors: np.ndarray = None) -> None:
        """
        Add a batch of experiences to the priority replay buffer, with
        priorities based on the TD errors.

        Parameters
        ----------
        batch : ExperiencesBatch
            A batch of experiences to be added to the buffer.
        td_errors : np.ndarray, optional
            The temporal difference errors for each experience in the batch
            (default is None).
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
        """
        Sample a batch of experiences from the priority replay buffer, with
        priority sampling.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        ExperiencesBatch
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
        """
        Update the priorities of experiences in the buffer based on new TD
        errors.

        Parameters
        ----------
        indices : np.ndarray
            The indices of the experiences whose priorities will be updated.
        td_errors : np.ndarray
            The new temporal difference errors used to update the priorities.
        """
        self.priorities[indices] = np.clip(
            td_errors, a_min=self.min_priority, a_max=None
        )
