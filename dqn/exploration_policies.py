"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


class ExplorationPolicy(ABC):
    """
    Abstract base class for exploration policies used with DQN agents.
    """

    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Selects an action based on the given Q-values.
        """
        pass

    @abstractmethod
    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Selects actions for a batch of Q-values.
        """
        pass

    @abstractmethod
    def update_params(self, steps: int = 1) -> None:
        """
        Updates the parameters of the exploration policy.
        """
        if steps < 1:
            raise ValueError("Steps must be greater than 1.")


class EpsilonGreedyPolicy(ExplorationPolicy):
    """
    Epsilon-greedy exploration policy for reinforcement learning.

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
        """
        Initializes the EpsilonGreedyPolicy.

        Parameters
        ----------
        epsilon : float, optional
            The initial exploration probability, default is 1.0.
        epsilon_min : float, optional
            The minimum value of epsilon, default is 0.01.
        epsilon_decay : float, optional
            The decay factor for epsilon, default is 0.9999.
        decay_type : {'exponential', 'linear', 'fixed'}, optional
            The type of decay for epsilon, default is 'exponential'.
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

    def select_action(self, q_values: np.ndarray) -> int:
        """
        Selects an action using epsilon-greedy strategy.

        Parameters
        ----------
        q_values : np.ndarray
            The Q-values for each action, with shape (num_actions,).

        Returns
        -------
        int
            The index of the selected action.
        """
        num_actions = q_values.size
        if np.random.rand() <= self.epsilon:
            return np.random.choice(num_actions)
        action = np.argmax(q_values)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Selects actions for a batch of Q-values using epsilon-greedy strategy.

        Parameters
        ----------
        q_values : np.ndarray
            The Q-values for each action, with shape (batch_size, num_actions).

        Returns
        -------
        np.ndarray
            An array of selected action indices, with shape (batch_size,).
        """
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
        """
        Updates the epsilon parameter based on the decay type.

        Parameters
        ----------
        steps : int, optional
            The number of steps to update epsilon, default is 1.

        Raises
        ------
        ValueError
            If an invalid decay type is provided.
        """
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


class BoltzmannPolicy(ExplorationPolicy):
    """
    Boltzmann exploration policy for reinforcement learning.

    This policy selects actions based on the softmax distribution of Q-values.
    """

    def __init__(self, tau=1.0) -> None:
        """
        Initializes the BoltzmannPolicy.

        Parameters
        ----------
        tau : float, optional
            The temperature parameter, default is 1.0.
        """
        self.tau = tau

    def select_action(self, q_values: np.ndarray) -> int:
        """
        Selects an action using the Boltzmann strategy.

        Parameters
        ----------
        q_values : np.ndarray
            The Q-values for each action, with shape (num_actions,).

        Returns
        -------
        int
            The index of the selected action.
        """
        num_actions = q_values.size
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(num_actions, p=probabilities)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Selects actions for a batch of Q-values using the Boltzmann strategy.

        Parameters
        ----------
        q_values : np.ndarray
            The Q-values for each action, with shape (batch_size, num_actions).

        Returns
        -------
        np.ndarray
            An array of selected action indices, with shape (batch_size,).
        """
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
        """
        Do nothing.

        Empty implementation of update params method from policy base class.

        Parameters
        ----------
        steps : int, optional
            The number of steps to update tau, default is 1.
        """
        return super().update_params(steps)
