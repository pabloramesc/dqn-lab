"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    """
    A class representing a single experience in a DQN agent training.
    """

    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool

    def to_tuple(self) -> tuple[np.ndarray, int, np.ndarray, float, bool]:
        """
        Convert the experience into a tuple format.

        Returns
        -------
        tuple
            A tuple containing (state, action, next_state, reward, done).
        """
        return (self.state, self.action, self.next_state, self.reward, self.done)


@dataclass
class ExperiencesBatch:
    """
    A class representing a batch of experiences, typically used for training.
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    indices: np.ndarray = None
    weights: np.ndarray = None

    @property
    def size(self) -> int:
        """
        The number of experiences in the batch, after checking consistency.
        """
        return self._check_consistency()

    def to_experiences(self) -> list[Experience]:
        """
        Convert the batch to a list of `Experience` objects.

        Returns
        -------
        list
            A list of `Experience` objects.
        """
        size = self._check_consistency()
        experiences = [None] * size
        for i in range(size):
            experiences[i] = Experience(
                state=self.states[i],
                action=self.actions[i],
                next_state=self.next_states[i],
                reward=self.rewards[i],
                done=self.dones[i],
            )
        return experiences

    def _check_consistency(self) -> int:
        """
        Ensure all arrays in the batch have consistent shapes and sizes.
        """
        if self.states.shape != self.next_states.shape:
            raise ValueError("States and next states shapes must be equal")

        size = self.states.shape[0]

        if self.actions.ndim > 1 or self.actions.shape[0] != size:
            raise ValueError(f"Actions must be a 1D array of size {size}")

        if self.rewards.ndim > 1 or self.rewards.shape[0] != size:
            raise ValueError(f"Rewards must be a 1D array of size {size}")

        if self.dones.ndim > 1 or self.dones.shape[0] != size:
            raise ValueError(f"Dones must be a 1D array of size {size}")

        if self.indices is None:
            return size

        if self.indices.ndim > 1 or self.indices.shape[0] != size:
            raise ValueError(f"Indices must be a 1D array of size {size}")

        if self.weights is None:
            return size

        if self.weights.ndim > 1 or self.weights.shape[0] != size:
            raise ValueError(f"Weights must be a 1D array of size {size}")

        return size
