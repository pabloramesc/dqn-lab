"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import keras as kr
import numpy as np
import tensorflow as tf

from .exploration_policies import EpsilonGreedyPolicy, ExplorationPolicy
from .buffers.replay_buffer import (
    Experience,
    ExperiencesBatch,
    PriorityReplayBuffer,
    ReplayBuffer,
)


class DQNAgent:
    """
    DQN Agent class for training a Deep Q-Network agent on a given environment.
    """

    def __init__(
        self,
        model: kr.Model = None,
        batch_size: int = 64,
        gamma: float = 0.95,
        policy: ExplorationPolicy = None,
        memory: ReplayBuffer = None,
        memory_size: int = 10_000,
    ) -> None:
        """
        Initializes the DQN Agent with the provided parameters.
        """
        self.model = model
        self.target_model: kr.Model = None
        if self.model is not None:
            self.set_model(model)

        self.memory = memory or ReplayBuffer(memory_size)
        if self.memory._max_size < batch_size:
            raise ValueError(
                f"Memory max size {self.memory._max_size} cannot be smaller than batch size {self.batch_size}"
            )

        self.batch_size = batch_size
        self.gamma = gamma

        self.policy = policy or EpsilonGreedyPolicy()

    def act(self, state: np.ndarray) -> int:
        """
        Chooses an action based on the current state using the exploration policy.
        """
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        action = self.policy.select_action(q_values[0])
        return action

    def act_on_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Chooses actions for a batch of states using the exploration policy.
        """
        if states.ndim == 3:
            states = np.expand_dims(states, axis=0)
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        q_values = self.model(states_tensor, training=False).numpy()
        actions = self.policy.select_action_batch(q_values)
        return actions

    def add_experience(self, exp: Experience) -> None:
        """
        Adds a single experience to the replay buffer.
        """
        self.memory.add(exp)

    def add_experiences_batch(self, batch: ExperiencesBatch) -> None:
        """
        Adds a batch of experiences to the replay buffer.
        """
        experiences = batch.to_experiences()
        self.memory.add_batch(experiences)

    def train(self) -> dict:
        """
        Performs a training step on the agent, using a batch of experiences from memory.
        """
        if self.memory.size < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, targets, weights = self._process_batch(batch)

        metrics = self.model.train_on_batch(
            states, targets, sample_weight=weights, return_dict=True
        )

        self.policy.update_params()

        return metrics

    def set_model(self, model: kr.Model) -> None:
        """
        Sets the model and target model for the agent.
        """
        self.model = model
        self.target_model = kr.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def update_target_model(self) -> None:
        """
        Updates the target model with the current weights of the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def _process_batch(
        self, batch: ExperiencesBatch
    ) -> tuple[tf.Tensor, tf.Tensor, np.ndarray]:
        """
        Processes a batch of experiences to prepare inputs for training.
        """
        states = tf.convert_to_tensor(batch.states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch.next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.dones, dtype=tf.bool)

        if isinstance(self.memory, PriorityReplayBuffer):
            targets, td_errors = compute_targets_per(
                self.model,
                self.target_model,
                states,
                next_states,
                actions,
                rewards,
                dones,
                self.gamma,
            )
            self.memory.update_priorities(batch.indices, td_errors.numpy())
            return states, targets, batch.weights

        targets = compute_targets(
            self.model,
            self.target_model,
            states,
            next_states,
            actions,
            rewards,
            dones,
            self.gamma,
        )
        return states, targets, None  # sample_weight = None


@tf.function(jit_compile=True)
def compute_targets(
    model: tf.keras.Model,
    target_model: tf.keras.Model,
    states: tf.Tensor,
    next_states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    dones: tf.Tensor,
    gamma: float,
) -> tf.Tensor:
    """
    TensorFlow helper function to compute Q-values targets for DQN.

    Parameters
    ----------
    model : tf.keras.Model
        The current Q-network model.
    target_model : tf.keras.Model
        The target Q-network model.
    states : tf.Tensor
        The states from the environment.
    next_states : tf.Tensor
        The next states from the environment.
    actions : tf.Tensor
        The actions taken in the states.
    rewards : tf.Tensor
        The rewards received for the actions taken.
    dones : tf.Tensor
        Whether each episode is done (True or False).
    gamma : float
        The discount factor for future rewards.

    Returns
    -------
    tf.Tensor
        The updated Q-values.
    """
    q_values = model(states, training=False)
    q_next = target_model(next_states, training=False)

    max_next_q = tf.reduce_max(q_next, axis=1)
    q_target = rewards + gamma * max_next_q * tf.cast(~dones, tf.float32)

    indices = tf.range(tf.shape(actions)[0])
    indices = tf.stack([indices, actions], axis=1)

    updated_q_values = tf.tensor_scatter_nd_update(q_values, indices, q_target)
    return updated_q_values


@tf.function(jit_compile=True)
def compute_targets_per(
    model: tf.keras.Model,
    target_model: tf.keras.Model,
    states: tf.Tensor,
    next_states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    dones: tf.Tensor,
    gamma: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow helper function to compute Q-values and TD-errors for PER
    (Prioritized Experience Replay).

    Parameters
    ----------
    model : tf.keras.Model
        The current Q-network model.
    target_model : tf.keras.Model
        The target Q-network model.
    states : tf.Tensor
        The states from the environment.
    next_states : tf.Tensor
        The next states from the environment.
    actions : tf.Tensor
        The actions taken in the states.
    rewards : tf.Tensor
        The rewards received for the actions taken.
    dones : tf.Tensor
        Whether each episode is done (True or False).
    gamma : float
        The discount factor for future rewards.

    Returns
    -------
    tuple
        - tf.Tensor: The updated Q-values.
        - tf.Tensor: The TD-errors for prioritized experience replay.
    """
    q_values = model(states, training=False)
    q_next = target_model(next_states, training=False)

    max_next_q = tf.reduce_max(q_next, axis=1)
    q_target = rewards + gamma * max_next_q * tf.cast(~dones, tf.float32)

    indices = tf.range(tf.shape(actions)[0])
    indices = tf.stack([indices, actions], axis=1)

    q_actual = tf.gather_nd(q_values, indices)
    td_errors = q_target - q_actual

    updated_q_values = tf.tensor_scatter_nd_update(q_values, indices, q_target)
    return updated_q_values, td_errors
