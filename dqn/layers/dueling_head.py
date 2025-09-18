import tensorflow as tf
import keras


class DuelingHead(keras.Layer):
    """
    Combine value and advantage streams into Q-values for Duealing DQN.

    Formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    """

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
        """
        Compute Q-values from value and advantage streams.

        Args:
            inputs: Tuple of tensors (value, advantage) where value has shape
                (batch_size, 1) and advantage has shape (batch_size, num_actions).

        Returns:
            Q-values, tensor of shape (batch_size, num_actions).
        """
        value, advantage = inputs
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - advantage_mean)
        return q_values
