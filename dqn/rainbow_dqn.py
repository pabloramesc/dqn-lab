from math import gamma
from typing import Optional

import keras
import tensorflow as tf

from .buffers import NStepPER
from .dqn_agent import DQNAgent
from .experiences import ExperiencesBatch
from .policies import ExplorationPolicy


class RainbowDQN(DQNAgent):
    def __init__(
        self,
        model: keras.Model,
        policy: Optional[ExplorationPolicy] = None,
        batch_size: int = 32,
        memory_size: int = 100_000,
        update_freq: int = 10_000,
        gamma: float = 0.99,
        n_step: int = 3,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            policy=policy,
            batch_size=batch_size,
            memory_size=memory_size,
            gamma=gamma,
            update_freq=update_freq,
        )
        self.memory = NStepPER(
            max_size=memory_size,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            beta_annealing=beta_annealing,
        )

    def _train_interface(self, batch: ExperiencesBatch) -> dict:
        loss, td_errors = self._train_step(
            states=tf.convert_to_tensor(batch.states, dtype=tf.float32),
            actions=tf.convert_to_tensor(batch.actions, dtype=tf.int32),
            next_states=tf.convert_to_tensor(batch.next_states, dtype=tf.float32),
            rewards=tf.convert_to_tensor(batch.rewards, dtype=tf.float32),
            dones=tf.convert_to_tensor(batch.dones, dtype=tf.bool),
            steps=tf.convert_to_tensor(batch.steps, dtype=tf.int32),
        )
        self.memory.update_priorities(batch.indices, td_errors)  # type: ignore
        return {"loss": loss.numpy().item()}

    @tf.function
    def _train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        next_states: tf.Tensor,
        rewards: tf.Tensor,
        dones: tf.Tensor,
        steps: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        
        # Compute target Q-values
        q_next = self.target_model(next_states, training=False)
        max_q_next = tf.reduce_max(q_next, axis=1)
        not_done_mask = tf.cast(tf.logical_not(dones), dtype=tf.float32)
        gamma_n = tf.pow(self.gamma, tf.cast(steps, dtype=tf.float32))
        q_target = rewards + gamma_n * max_q_next * not_done_mask
        
        # Compute loss insde gradient tape
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            batch_indices = tf.range(actions.shape[0])
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_actual = tf.gather_nd(q_values, action_indices)

            td_errors = q_target - q_actual
            huber = tf.keras.losses.Huber()
            loss = huber(q_target, q_actual)

        # Apply gradients
        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)
        grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
        self.model.optimizer.apply_gradients(zip(grads, vars))

        return loss, td_errors
