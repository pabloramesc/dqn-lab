import numpy as np

from .base import ExplorationPolicy


class BoltzmannPolicy(ExplorationPolicy):
    """Boltzmann exploration policy for reinforcement learning.

    This policy selects actions based on the softmax distribution of Q-values.
    """

    def __init__(self, tau=1.0) -> None:
        """
        Initializes the BoltzmannPolicy.

        Args:
            tau: The temperature parameter.
        """
        self.tau = tau

    def select_action(self, q_values: np.ndarray) -> int:
        num_actions = q_values.size
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
        """Do nothing. Boltzmann policy doesn't use any dynamic parameter."""
        return super().update_params(steps)
