"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

import tensorflow as tf
import keras.api as kr
import numpy as np

from dqn.exploration_policies import EpsilonGreedyPolicy, ExplorationPolicy
from dqn.replay_buffers import (
    Experience,
    ExperiencesBatch,
    PriorityReplayBuffer,
    ReplayBuffer,
)


class DQNAgent:
    """
    DQN Agent class for training a Deep Q-Network agent on a given environment.

    This class encapsulates the logic for interacting with an environment,
    storing experiences, training a neural network model, and implementing
    policies for exploration and exploitation. It supports features like replay
    buffer, target model updates, and autosave.
    """

    def __init__(
        self,
        model: kr.Model = None,
        batch_size: int = 64,
        gamma: float = 0.95,
        policy: ExplorationPolicy = None,
        memory: ReplayBuffer = None,
        memory_size: int = 10_000,
        update_steps: int = 1000,
        autosave_steps: int = 0,
        file_name: str = None,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the DQN Agent with the provided parameters.

        Parameters
        ----------
        model : kr.Model, optional
            The model to be used for Q-value prediction (default is None).
        batch_size : int, optional
            The batch size used for training (default is 64).
        gamma : float, optional
            Discount factor for future rewards (default is 0.95).
        policy : ExplorationPolicy, optional
            Exploration policy to be used by the agent
            (default is EpsilonGreedyPolicy).
        memory : ReplayBuffer, optional
            Replay buffer to store experiences
            (default is a new ReplayBuffer).
        memory_size : int, optional
            Maximum size of the replay buffer (default is 10,000).
        update_steps : int, optional
            Number of steps between updates of the target model
            (default is 1000).
        autosave_steps : int, optional
            Number of steps between saving the model (default is 0).
        file_name : str, optional
            The file name for saving/loading the model (default is None).
        verbose : bool, optional
            If True, will print information during training (default is True).
        """
        self.model = model
        self.batch_size = batch_size
        self.gamma = gamma

        self.target_model: kr.Model = None
        if self.model is not None:
            self.set_model(model)

        self.memory: ReplayBuffer = (
            memory if memory is not None else ReplayBuffer(memory_size)
        )

        if self.memory.max_size < batch_size:
            raise ValueError(
                f"Memory max size {self.memory.max_size} cannot be smaller than batch size {self.batch_size}"
            )

        self.policy: ExplorationPolicy = (
            policy if policy is not None else EpsilonGreedyPolicy()
        )

        self.update_steps = update_steps
        self.autosave_steps = autosave_steps
        self.file_name = file_name
        self.verbose = verbose

        self.train_steps = int(0)
        self.train_t0: float = None

    @property
    def train_elapsed(self) -> float:
        """
        The elapsed time since the first training step
        """
        if self.train_t0 is None:
            return None
        return time.time() - self.train_t0

    @property
    def train_speed(self) -> float:
        """
        The training speed in steps per second calculated as
        `train_steps / train_elapsed`.
        """
        if self.train_t0 is None:
            return None
        return self.train_steps / self.train_elapsed

    def act(self, state: np.ndarray) -> int:
        """
        Chooses an action based on the current state using the exploration
        policy.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        int
            The action to be taken.
        """
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        action = self.policy.select_action(q_values[0])
        return action

    def act_on_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Chooses actions for a batch of states using the exploration policy.

        Parameters
        ----------
        states : np.ndarray
            A batch of states to choose actions for.

        Returns
        -------
        np.ndarray
            An array of actions for each state in the batch.
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

        Parameters
        ----------
        exp : Experience
            The experience to be added to the replay buffer.
        """
        self.memory.add(exp)

    def add_experiences_batch(self, batch: ExperiencesBatch) -> None:
        """
        Adds a batch of experiences to the replay buffer.

        Parameters
        ----------
        batch : ExperiencesBatch
            The batch of experiences to be added to the replay buffer.
        """
        experiences = batch.to_experiences()
        self.memory.add_batch(experiences)

    def train(self) -> dict:
        """
        Performs a training step on the agent, using a batch of experiences
        from memory.

        Returns
        -------
        dict
            The training metrics returned by the model's `train_on_batch`
            method. If the memory size is smaller than batch it returns None.
        """
        if self.memory.size < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, targets, weights = self._process_batch(batch)

        metrics = self.model.train_on_batch(
            states, targets, sample_weight=weights, return_dict=True
        )

        self._post_train_update(steps=1)
        return metrics

    def set_model(self, model: kr.Model) -> None:
        """
        Sets the model and target model for the agent.

        Parameters
        ----------
        model : kr.Model
            The model to be used by the agent.
        """
        self.model = model
        self.target_model = kr.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def update_target_model(self) -> None:
        """
        Updates the target model with the current weights of the main model.
        """
        self.target_model.set_weights(self.model.get_weights())
        if self.verbose:
            print("DQN Agent: Target model updated.")

    def save_model(self, file_name: str = None) -> None:
        """
        Saves the current model to the specified file name.

        Parameters
        ----------
        file_name : str, optional
            The file name where the model will be saved. If None, the default
            model file name from the agent's initialization is used.
        """
        file_name = file_name or self.file_name
        self.model.save(file_name)
        if self.verbose:
            print(f"DQN Agent: Model saved to '{file_name}'.")

    def load_model(self, file_name: str = None, compile: bool = True) -> None:
        """
        Loads a model from the specified file name.

         Parameters
        ----------
        file_name : str, optional
            The file name from which to load the model. If None, the default
            file name from the agent's initialization is used.
        compile : bool, optional
            Whether to compile the model after loading. Default is True.
        """
        file_name = file_name or self.file_name
        model = kr.models.load_model(file_name, compile=compile)
        self.set_model(model)
        if self.verbose:
            print(f"DQN Agent: Model loaded from '{file_name}'.")

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

    def _post_train_update(self, steps: int = 1):
        """
        Performs updates after each training step, including updating the
        target model and optionally saving the model.
        """
        self.policy.update_params()
        self.train_steps += steps

        if self.train_t0 is None:
            self.train_t0 = time.time()

        if self.update_steps > 0 and self.train_steps % self.update_steps == 0:
            self.update_target_model()

        if self.autosave_steps > 0 and self.train_steps % self.autosave_steps == 0:
            self.save_model()


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
