import numpy as np

from .base import ExplorationPolicy

from typing import Literal


class BoltzmannPolicy(ExplorationPolicy):
    """Boltzmann exploration policy for reinforcement learning.

    This policy selects actions based on the softmax distribution of Q-values.
    """

    def __init__(
        self,
        tau: float = 1.0,
        tau_min: float = 0.0,
        tau_decay: float = 1e-6,
        decay_type: Literal["fixed", "linear", "exponential"] = "fixed",
    ) -> None:
        """
        Initializes the BoltzmannPolicy.

        Args:
            tau: The temperature parameter.
            tau_min: The minimum value of tau.
            tau_decay: The decay factor for tau.
            decay_type: The type of decay {'fixed', 'linear', 'exponential'}.
        """
        self.tau = tau
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.decay_type = decay_type

    def select_action(self, q_values: np.ndarray) -> int:
        num_actions = q_values.size

        # Pure exploitation: greedy action
        if self.tau <= 0.0:
            return np.argmax(q_values).item()

        # Softmax distribution
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(num_actions, p=probabilities)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        batch_size = q_values.shape[0]
        num_actions = q_values.shape[1]
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        actions = np.array(
            [
                np.random.choice(num_actions, p=probabilities[i])
                for i in range(batch_size)
            ]
        )
        return actions

    def update_params(self, steps: int = 1) -> None:
        super().update_params(steps)

        if self.decay_type == "fixed":
            return

        elif self.decay_type == "linear":
            self.tau = max(self.tau, self.tau - self.tau_decay * steps)

        elif self.decay_type == "exponential":
            self.tau = max(self.tau, self.tau * self.tau_decay**steps)

        else:
            raise ValueError(f"Not valid decay type '{self.decay_type}'.")
        
    def get_dynamic_params(self) -> dict[str, float]:
        return {"tau": self.tau}

    def set_full_exploration(self) -> None:
        """Force uniform random actions."""
        self.tau = np.inf

    def set_full_exploitation(self) -> None:
        """Force pure exploitation: greedy (argmax) selection."""
        self.tau = 0.0
