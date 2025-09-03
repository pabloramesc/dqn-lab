from typing import Literal

import numpy as np

from .base import ExplorationPolicy


class EpsilonGreedyPolicy(ExplorationPolicy):
    """Epsilon-greedy exploration policy for reinforcement learning.

    This policy selects a random action with probability epsilon (exploration).
    Epsilon values near 1.0 promote exploration, while values near 0.0 select
    best actions according to Q-values (exploitation).
    """

    def __init__(
        self,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999,
        decay_type: Literal["exponential", "linear", "fixed"] = "exponential",
    ) -> None:
        """Initializes the epsilon-greedy policy.

        Args:
            epsilon: The initial exploration probability.
            epsilon_min: The minimum value of epsilon.
            epsilon_decay: The decay factor for epsilon.
            decay_type: The type of decay {'exponential', 'linear', 'fixed'}.
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

    def select_action(self, q_values: np.ndarray) -> int:
        num_actions = q_values.size
        if np.random.rand() <= self.epsilon:
            return np.random.choice(num_actions)
        action = np.argmax(q_values)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        batch_size = q_values.shape[0]
        num_actions = q_values.shape[1]
        # Exploration: random actions
        random_actions = np.random.choice(num_actions, batch_size)
        # Exploitation: predicted actions
        greedy_actions = np.argmax(q_values, axis=1)
        # Epsilon-greedy policy
        mask = np.random.rand(batch_size) <= self.epsilon
        actions = np.where(mask, random_actions, greedy_actions)
        return actions

    def update_params(self, steps: int = 1) -> None:
        super().update_params(steps)
        if self.decay_type == "exponential":
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay**steps
            )
        elif self.decay_type == "linear":
            self.epsilon = max(
                self.epsilon_min, self.epsilon - self.epsilon_decay * steps
            )
        elif self.decay_type == "fixed":
            return  # No update for fixed epsilon
        else:
            raise ValueError(
                f"Not valid decay type '{self.decay_type}'. Valid types are 'exponential', 'linear' or 'fixed'."
            )