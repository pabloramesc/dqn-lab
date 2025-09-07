"""Abstract base class for exploration policies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ExplorationPolicy(ABC):
    """Abstract base class for exploration policies used with DQN agents."""

    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        """Select action based on given Q-values.

        Args:
            q_values: Array of Q-values with shape (num_actions,).

        Returns:
            Index of the selected action.
        """
        pass

    @abstractmethod
    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """Select actions for a batch of Q-values.

        Args:
            q_values: Array of Q-values with shape (batch_size, num_actions).

        Returns:
            Array of selected indices with shape (batch_size,).
        """
        pass

    @abstractmethod
    def update_params(self, steps: int = 1) -> None:
        """Updates the parameters of the exploration policy.

        Args:
            steps: Number of steps to update.

        Raises:
            ValuerError: If steps < 1.
        """
        if steps < 1:
            raise ValueError("Steps must be greater than 0.")

    def get_dynamic_params(self) -> dict[str, Any]:
        """Return a dictionary with policy dynamic parameters."""
        return {}

    def set_full_exploration(self) -> None:
        """Force the policy into pure exploration mode (actions 100% random)."""
        return
    
    def set_full_exploitation(self) -> None:
        """Force the polciy into pure exploitation mode (no exploration)."""
        return