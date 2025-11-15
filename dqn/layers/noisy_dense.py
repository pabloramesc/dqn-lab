import keras
import numpy as np
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="dqn.layers")
class NoisyDense(layers.Layer):
    def __init__(self, units, activation=None, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.sigma_init = sigma_init

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        # Base weights and biases
        self.mu_w = self.add_weight(
            name="mu_w",
            shape=[self.input_dim, self.units],
            initializer=keras.initializers.RandomUniform(
                -1 / np.sqrt(self.input_dim), 1 / np.sqrt(self.input_dim)
            ),
            trainable=True,
        )
        self.sigma_w = self.add_weight(
            name="sigma_w",
            shape=[self.input_dim, self.units],
            initializer=keras.initializers.Constant(self.sigma_init),
            trainable=True,
        )
        self.mu_b = self.add_weight(
            name="mu_b",
            shape=[self.units],
            initializer=keras.initializers.RandomUniform(
                -1 / np.sqrt(self.input_dim), 1 / np.sqrt(self.input_dim)
            ),
            trainable=True,
        )
        self.sigma_b = self.add_weight(
            name="sigma_b",
            shape=[self.units],
            initializer=keras.initializers.Constant(self.sigma_init),
            trainable=True,
        )

    def call(self, inputs, training=True):
        if training:
            # Factorized Gaussian noise (Fortunato et al., 2018)
            epsilon_in = tf.random.normal([self.input_dim, 1])
            epsilon_out = tf.random.normal([1, self.units])
            f_in = tf.sign(epsilon_in) * tf.sqrt(tf.abs(epsilon_in))
            f_out = tf.sign(epsilon_out) * tf.sqrt(tf.abs(epsilon_out))
            epsilon_w = f_in * f_out
            epsilon_b = tf.squeeze(f_out)

            w = self.mu_w + self.sigma_w * epsilon_w
            b = self.mu_b + self.sigma_b * epsilon_b
        else:
            # Deterministic mode (for evaluation)
            w, b = self.mu_w, self.mu_b

        return self.activation(tf.matmul(inputs, w) + b)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "sigma_init": self.sigma_init,
            }
        )
        return config
