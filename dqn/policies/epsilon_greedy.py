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
        epsilon: float = 1.0,
        epsilon_min: float = 0.0,
        epsilon_decay: float = 1e-6,
        decay_type: Literal["fixed", "linear", "exponential"] = "fixed",
    ) -> None:
        """Initializes the epsilon-greedy policy.

        Args:
            epsilon: The initial exploration probability.
            epsilon_min: The minimum value of epsilon.
            epsilon_decay: The decay factor for epsilon.
            decay_type: The type of decay {'fixed', 'linear', 'exponential'}.
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
        return action.item()

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        batch_size, num_actions = q_values.shape
        
        # Exploration mask
        mask = np.random.rand(batch_size) <= self.epsilon
        
        # Initialize actions array
        actions = np.empty(batch_size, dtype=np.int32)

        # Exploration: random actions
        actions[mask] = np.random.randint(num_actions, size=mask.sum())

        # Exploitation: greedy actions
        actions[~mask] = np.argmax(q_values[~mask], axis=1)

        return actions

    def update_params(self, steps: int = 1) -> None:
        super().update_params(steps)

        if self.decay_type == "fixed":
            return  # No update for fixed epsilon

        elif self.decay_type == "linear":
            self.epsilon = max(
                self.epsilon_min, self.epsilon - self.epsilon_decay * steps
            )

        elif self.decay_type == "exponential":
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay**steps
            )

        else:
            raise ValueError(f"Not valid decay type '{self.decay_type}'.")

    def get_dynamic_params(self) -> dict[str, float]:
        return {"epsilon": self.epsilon}

    def set_full_exploration(self) -> None:
        """Force uniform random actions."""
        self.decay_type = "fixed"
        self.epsilon = 1.0

    def set_full_exploitation(self) -> None:
        """Force pure exploitation: greedy (argmax) selection."""
        self.decay_type = "fixed"
        self.epsilon = 0.0
