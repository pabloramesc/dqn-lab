# %%
# 🕹️ DQN for Atari Breakout with VGG-style CNN, dueling Q-network,
# prioritized experience replay (PER), and vectorized environments.


# %%
# 📦 Import modules and setup environment
import os
import ale_py
import gymnasium as gym
import keras
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
# 🧠 Dueling DQN with VGG-style convolutional layers
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    Rescaling,
)
from keras.losses import Huber
from keras.models import Model
from keras.optimizers import Adam

from dqn.layers import DuelingHead


def create_model(state_shape: tuple[int, ...], num_actions: int) -> Model:
    inputs = Input(shape=state_shape, dtype="uint8")

    # Normalize inputs
    x = Rescaling(1.0 / 255.0)(inputs)

    x = Conv2D(32, 8, strides=4, activation="relu")(x)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)

    x = Flatten()(x)

    # Value head
    v = Dense(512, activation="relu")(x)
    v = Dense(1, activation="linear")(v)

    # Advantage head
    a = Dense(512, activation="relu")(x)
    a = Dense(num_actions, activation="linear")(a)

    # Combine value and advantage: V(s) + A(s, a) -> Q(s, a)
    q = DuelingHead(dtype="float32")([v, a])

    model = Model(inputs=inputs, outputs=q)
    model.compile(optimizer=Adam(learning_rate=0.0000625, epsilon=1.5e-4), loss=Huber(delta=1.0))  # type: ignore
    return model


# %%
# 🤖 Configure the DQN agent with PER (Prioritized Replay Buffer)
from dqn import RainbowDQN, EpsilonGreedyPolicy
from dqn.wrappers import AtariWrapper
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


# Env settings
def make_env():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariWrapper(env)
    return env


num_envs = 4  # 🚀 Run 4 parallel environments
envs = AsyncVectorEnv([make_env for _ in range(num_envs)])

state_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n  # type: ignore

# Create the DQN agent
model = create_model(state_shape, num_actions)  # type: ignore
policy = EpsilonGreedyPolicy(epsilon_min=0.1, epsilon_decay=1e-5, decay_type="linear")
agent = RainbowDQN(
    model=model,
    policy=policy,
    batch_size=32,
    memory_size=500_000,
    update_freq=10_000,
    gamma=0.99,
    n_step=3,
    alpha=0.6,
    beta=0.4,
    beta_annealing=0.0,
)


# Load pre-trained model if it exists
model_path = "models/breakout-rainbow-dqn.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(
        filepath=model_path, custom_objects={"DuelingHead": DuelingHead}, compile=True
    )  # Use custom objects to deserialize DuelinHead custom layer
    agent.set_model(model)  # type: ignore
    policy.epsilon = 1.0  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()  # type: ignore


# %%
# 💪 Training using vectorized environments (faster ⚡)
agent.learn_parallel(
    envs, # type: ignore
    max_episodes=1_000_000,
    min_memory=50_000,
    train_every=1,  # train every step
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
