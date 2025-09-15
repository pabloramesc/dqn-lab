import keras
import numpy as np
import tensorflow as tf

from .buffers import CircularBuffer
from .experiences import Experience, ExperiencesBatch
from .policies import ExplorationPolicy


class DQNAgent:
    """Deep Q-Network (DQN) agent for reinforcement learning.

    This agent uses a neural network to approximate the Q-function and can
    interact with an environment using an exploration policy. Supports training
    with a replay buffer.
    """

    def __init__(
        self,
        model: keras.Model,
        policy: ExplorationPolicy,
        batch_size: int = 32,
        memory_size: int = 100_000,
        gamma: float = 0.99,
        update_freq: int = 10_000,
    ) -> None:
        """Initializes the DQN agent.

        Args:
            model: Keras model used to approximate the Q-function.
            policy: Exploration policy to use.
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
        self.policy = policy
        self.batch_size = int(batch_size)
        self.memory = CircularBuffer(memory_size)
        self.gamma = np.float32(gamma)
        self.update_freq = int(update_freq)

        self.train_steps = int(0)

    def set_model(self, model: keras.Model) -> None:
        """Sets the model for main and target networks.

        Args:
            model: Keras model to use as the Q-network.
        """
        self.model = model
        self.target_model = keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """Selects an action for a single state using the current policy.

        Args:
            state: Array representing the current state, with shape compatible with model input.

        Returns:
            Index of the selected action.
        """
        q_values = self.model(state[None, ...], training=False).numpy()
        action = self.policy.select_action(q_values[0])
        return action

    def act_on_batch(self, states: np.ndarray) -> np.ndarray:
        """Selects actions for a batch of states using the current policy.

        Args:
            states: Array of shape (batch_size, ...) representing multiple states.

        Returns:
            Array of selected action indices with shape (batch_size,).
        """
        q_values = self.model(states, training=False).numpy()
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

        return metrics

    def _train_interface(self, batch: ExperiencesBatch) -> dict:
        """Interface method to compute targets and perform train on batch."""
        # q_values, td_errors = self._compute_targets(batch)
        q_values, td_errors = self._compute_targets_optimized(batch)
        metrics = self.model.train_on_batch(batch.states, q_values, return_dict=True)
        return metrics

    def _compute_targets(
        self, batch: ExperiencesBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy and Keras target computation."""
        q_values = self.model(batch.states, training=False).numpy()
        q_next = self.target_model(batch.next_states, training=False).numpy()

        # Bellman equation
        max_next_q = np.max(q_next, axis=1)
        q_target = batch.rewards + self.gamma * max_next_q * (~batch.dones)

        # Update TD errors and PER buffer
        idx = np.arange(batch.size)
        td_errors = q_target - q_values[idx, batch.actions]

        # Update Q-values (after TD errors calculation)
        q_values[idx, batch.actions] = q_target

        return q_values, td_errors

    def _compute_targets_optimized(
        self, batch: ExperiencesBatch
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """TensorFlow optimized target computation."""
        q_values, td_errors = self._compute_targets_helper(
            batch.states,
            batch.actions,
            batch.next_states,
            batch.rewards,
            batch.dones,
        )  # type: ignore
        return q_values, td_errors

    @tf.function
    def _compute_targets_helper(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        next_states: tf.Tensor,
        rewards: tf.Tensor,
        dones: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        q_values = self.model(states, training=False)
        q_next = self.target_model(next_states, training=False)

        max_q_next = tf.reduce_max(q_next, axis=1)
        mask = tf.cast(tf.logical_not(dones), dtype=np.float32)
        q_target = rewards + self.gamma * max_q_next * mask

        indices = tf.range(actions.shape[0])
        indices = tf.stack([indices, actions], axis=1)

        q_actual = tf.gather_nd(q_values, indices)
        td_errors = q_target - q_actual

        q_values = tf.tensor_scatter_nd_update(q_values, indices, q_target)
        return q_values, td_errors
