from typing import Optional

import keras
import numpy as np
import tensorflow as tf

from .buffers import ReplayBuffer
from .experiences import Experience, ExperiencesBatch
from .policies import EpsilonGreedyPolicy, ExplorationPolicy
from .training import GymEnv, VectEnv, evaluate_agent, train_agent, train_parallel


class DQNAgent:
    """Deep Q-Network (DQN) agent for reinforcement learning.

    This agent uses a neural network to approximate the Q-function and can
    interact with an environment using an exploration policy. Supports training
    with a replay buffer.
    """

    def __init__(
        self,
        model: keras.Model,
        policy: Optional[ExplorationPolicy] = None,
        batch_size: int = 32,
        memory_size: int = 100_000,
        gamma: float = 0.99,
        update_freq: int = 10_000,
        clipnorm: float = 10.0,
    ) -> None:
        """Initializes the DQN agent.

        Args:
            model: Keras model used to approximate the Q-function.
            policy: Exploration policy to use. Default is epsilon-greedy policy
                with linear decay from 1.0 to 0.01 over 1M updates.
            batch_size: Number of experiences per training batch.
            memory_size: Maximum number of experiences to store in the replay buffer.
            gamma: Discount factor for future rewards.
            update_freq: Number of training steps between automatic target network updates.
        """
        if memory_size < batch_size:
            raise ValueError(
                f"Memory size ({memory_size}) must be greater than batch size ({batch_size})."
            )

        self.set_model(model)
        self.policy = policy or EpsilonGreedyPolicy(
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-6, decay_type="fixed"
        )
        self.batch_size = int(batch_size)
        self.memory = ReplayBuffer(memory_size)
        self.gamma = np.float32(gamma)
        self.update_freq = int(update_freq)
        self.clipnorm = np.float32(clipnorm)

        self.train_steps = int(0)

    def set_model(self, model: keras.Model) -> None:
        """Sets the model for main and target networks.

        Args:
            model: Keras model to use as the Q-network.
        """
        self.model = model
        self.target_model = keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    @profile
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Selects an action for a single state using the current policy.

        Args:
            state: Array representing the current state, with shape compatible with model input.

        Returns:
            Index of the selected action.
        """
        q_values = self.model(state[None, ...], training=training).numpy()
        action = self.policy.select_action(q_values[0])
        return action

    def act_on_batch(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """Selects actions for a batch of states using the current policy.

        Args:
            states: Array of shape (batch_size, ...) representing multiple states.

        Returns:
            Array of selected action indices with shape (batch_size,).
        """
        q_values = self.model(states, training=training).numpy()
        actions = self.policy.select_action_batch(q_values)
        return actions

    def add_experience(self, exp: Experience) -> None:
        """Adds a single experience to the replay buffer.

        Args:
            exp: Experience dataclass.
        """
        self.memory.add(exp)

    def add_experiences_batch(self, batch: ExperiencesBatch) -> None:
        """Adds a batch of experiences to the replay buffer.

        Args:
            batch: ExperiencesBatch containing multiple experiences.
        """
        self.memory.add_batch(batch)

    def update_target_model(self) -> None:
        """Updates the target network with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())

    @profile
    def train(self) -> dict | None:
        """Performs a single training step if enough experiences are available,
        and update target network based on the update frequency. Increments train steps counter.

        Returns:
            Dictionary of training metrics from the model, or None if
            there are not enough experiences in the replay buffer.
        """
        if self.memory.size < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        metrics = self._train_interface(batch)

        self.train_steps += 1
        if (
            self.update_freq > 0
            and self.train_steps > 0
            and self.train_steps % self.update_freq == 0
        ):
            self.update_target_model()

        self.policy.update_params()

        # metrics["memory_size"] = self.memory.size
        # metrics["train_steps"] = self.train_steps
        # metrics.update(self.policy.dynamic_params)
        return metrics

    # @profile
    def _train_interface(self, batch: ExperiencesBatch) -> dict:
        # Convert batch numpy arrays to tensors
        states = tf.convert_to_tensor(batch.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
        next_states = tf.convert_to_tensor(batch.next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.dones, dtype=tf.bool)

        # Call the training step (optimized TensorFlow function)
        loss = self._train_step(states, actions, next_states, rewards, dones)

        # Return training metrics
        return {"loss": loss}

    @tf.function(jit_compile=True)
    def _train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        next_states: tf.Tensor,
        rewards: tf.Tensor,
        dones: tf.Tensor,
    ) -> tf.Tensor:

        # Compute target Q-values
        q_next = self.target_model(next_states, training=False)
        max_q_next = tf.reduce_max(q_next, axis=1)
        not_done_mask = tf.cast(tf.logical_not(dones), dtype=tf.float32)
        q_target = rewards + self.gamma * max_q_next * not_done_mask
        
        # Prepare indices to gather the Q-values for taken actions
        batch_indices = tf.range(actions.shape[0])
        action_indices = tf.stack([batch_indices, actions], axis=1)

        # Compute loss inside gradient tape
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            q_actual = tf.gather_nd(q_values, action_indices)

            huber = tf.keras.losses.Huber()
            loss = huber(q_target, q_actual)

        # Apply gradients
        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)
        grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
        self.model.optimizer.apply_gradients(zip(grads, vars))

        return loss

    def evaluate(self, env: GymEnv, **kwargs):
        return evaluate_agent(env=env, agent=self, render=True, verbose=True, **kwargs)

    def learn(self, env: GymEnv, **kwargs):
        return train_agent(env=env, agent=self, **kwargs)

    def learn_parallel(self, envs: VectEnv, **kwargs):
        return train_parallel(envs=envs, agent=self, **kwargs)
