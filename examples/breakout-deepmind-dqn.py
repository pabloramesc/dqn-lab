# %%
# 🕹️ DQN Agent for Atari Breakout in Google DeepMind style

# %%
# 📦 Import modules and setup environment
import os
from typing import cast

import ale_py
import gymnasium as gym
import keras

# Ensure ALE environments are registered
gym.register_envs(ale_py)


# %%
# 🧠 Define the DQN model with CNN (DeepMind style)
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Rescaling
from keras.optimizers import Adam
from keras.losses import Huber

# Ensure keras image format is (height, width, channels) <-- applies to Conv2D layers
from keras.backend import set_image_data_format

set_image_data_format("channels_last")


def create_model(state_shape: tuple, num_actions: int) -> Model:
    model = Sequential(
        [
            InputLayer(shape=state_shape, dtype="uint8"),
            Rescaling(1.0 / 255.0),
            Conv2D(32, 8, strides=4, activation="relu"),
            Conv2D(64, 4, strides=2, activation="relu"),
            Conv2D(64, 3, strides=1, activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(num_actions, activation="linear"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.00025, clipnorm=1.0),  # type: ignore
        loss=Huber(delta=1.0),
    )

    return model


# %%
# 🤖 Initialize the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy

# Initialize state and action space dimensions
env = gym.make("ALE/Breakout-v5")
state_shape = (84, 84, 4)
num_actions = env.action_space.n  # type: ignore

# Create the DQN agent
model = create_model(state_shape, num_actions)
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.01, epsilon_decay=1e-6)
agent = DQNAgent(
    model=model,
    batch_size=32,
    gamma=0.99,
    memory_size=200_000,
    policy=policy,
    update_freq=10_000,
)

# Load pre-trained model if it exists
model_path = "models/breakout-deepmind-model.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(filepath=model_path, compile=True)
    model = cast(Model, model)
    agent.set_model(model)
    policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()


# %%
# 💪 Train the agent
from dqn.atari_utils import AtariTrainer

trainer = AtariTrainer(env=env, agent=agent, stack_frames=4)

trainer.train(
    max_episodes=1_000_000,
    max_noop_steps=30,
    min_memory_size=10_000,
    train_after_steps=4,
    max_episode_steps=100_000,
    max_score=500,
    model_path=model_path,
    verbose=True,
)


# Save the model
agent.model.save(filepath=model_path)
print(f"💾 Model saved to '{model_path}'.")


# %%
# 🧪 Test the trained agent
env = gym.make("ALE/Breakout-v5", render_mode="human")

trainer = AtariTrainer(env=env, agent=agent, stack_frames=None)

trainer.test(render=True)

# %% Run all cells above
