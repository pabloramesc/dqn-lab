# %%
# 🕹️ DQN Agent for Atari Breakout in Google DeepMind style

# %%
# 📦 Import modules and setup environment
import os
import ale_py
import gymnasium as gym
import keras as keras
from keras import backend, mixed_precision

# Ensure ALE environments are registered
gym.register_envs(ale_py)

# Set keras global policy to mixed_float16
mixed_precision.set_global_policy("mixed_float16")
print("Compute dtype:", mixed_precision.global_policy().compute_dtype)
print("Variable dtype:", mixed_precision.global_policy().variable_dtype)

# Ensure keras image format is (height, width, channels) - applies to Conv2D layers
backend.set_image_data_format("channels_last")
print("Image data format:", backend.image_data_format())


# %%
# 🧠 Define the DQN model with CNN (DeepMind style)
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Rescaling
from keras.losses import Huber
from keras.optimizers import Adam


def create_model(state_shape: tuple[int, ...], num_actions: int) -> Model:
    model = Sequential(
        [
            InputLayer(shape=state_shape, dtype="uint8"),
            Rescaling(1.0 / 255.0),
            Conv2D(32, 8, strides=4, activation="relu"),
            Conv2D(64, 4, strides=2, activation="relu"),
            Conv2D(64, 3, strides=1, activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(num_actions, activation="linear", dtype="float32"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.00025),  # type: ignore
        loss=Huber(delta=1.0),
    )

    return model


# %%
# 🤖 Initialize the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy
from dqn.wrappers import AtariWrapper

# Initialize environment state and action space dimensions
env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
env = AtariWrapper(env)

state_shape = env.observation_space.shape
num_actions = env.action_space.n  # type: ignore

# Create the DQN agent
model = create_model(state_shape, num_actions)  # type: ignore
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-6)
agent = DQNAgent(
    model=model,
    batch_size=32,
    gamma=0.99,
    memory_size=300_000,  # aprox. 5GB of RAM per 100k samples
    policy=policy,
    update_freq=10_000,
)

# Load pre-trained model if it exists
model_path = "models/breakout-vanilla-dqn.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(filepath=model_path, compile=True)
    agent.set_model(model)  # type: ignore
    policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()  # type: ignore


# %%
# 💪 Train the agent
agent.learn(
    env=env,  # type: ignore
    max_episodes=1_000_000,
    min_memory=50_000,
    train_every=4,
    max_score=1000,
    model_path=model_path,
    autosave_freq=10_000,
    verbose=True,
)

# Save the model
agent.model.save(filepath=model_path)
print(f"💾 Model saved to '{model_path}'.")


# %%
# 🧪 Test the trained agent
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env)

agent.evaluate(env)  # type: ignore


# %% Run all cells above
