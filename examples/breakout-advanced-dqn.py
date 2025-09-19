# %%
# 🕹️ DQN for Atari Breakout with VGG-style CNN, dueling Q-network,
# prioritized experience replay (PER), and vectorized environments.


# %%
# 📦 Import modules and setup environment
import os
import time

import ale_py
import gymnasium as gym
import keras
import numpy as np
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
    MaxPooling2D,
    Rescaling,
    Dropout,
    SpatialDropout2D,
)
from keras.losses import Huber
from keras.models import Model
from keras.optimizers import Adam

from dqn.layers import DuelingHead


def create_model(state_shape: tuple, num_actions: int) -> Model:
    inputs = Input(shape=state_shape, dtype="uint8")

    # Normalize inputs
    x = Rescaling(1.0 / 255.0)(inputs)

    # VGG-style convolutional layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.1)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.1)(x)

    x = Flatten()(x)

    # Value head
    v = Dense(512, activation="swish")(x)
    v = Dropout(0.2)(v)
    v = Dense(1, activation="linear")(v)

    # Advantage head
    a = Dense(512, activation="swish")(x)
    a = Dropout(0.2)(a)
    a = Dense(num_actions, activation="linear")(a)

    # Combine value and advantage: V(s) + A(s, a) -> Q(s, a)
    q = DuelingHead(dtype="float32")([v, a])

    model = Model(inputs=inputs, outputs=q)
    model.compile(optimizer=Adam(learning_rate=0.00025), loss=Huber(delta=1.0))  # type: ignore
    return model


# %%
# 🤖 Configure the DQN agent with PER (Prioritized Replay Buffer)
from dqn import DQNAgentPER, EpsilonGreedyPolicy, ExperiencesBatch

# Env settings
env = gym.make("ALE/Breakout-v5")
state_shape = (84, 84, 4)
num_actions = env.action_space.n  # type: ignore

# Create the DQN agent
model = create_model(state_shape, num_actions)
policy = EpsilonGreedyPolicy(
    epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-5, decay_type="linear"
)
agent = DQNAgentPER(
    model=model,
    policy=policy,
    batch_size=32,
    memory_size=200_000,
    gamma=0.99,
    update_freq=10_000,
    alpha=0.6,
    beta=0.4,
    beta_annealing=0.0,
)


# Load pre-trained model if it exists
model_path = "models/breakout-advanced-model.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(
        filepath=model_path, custom_objects={"DuelingHead": DuelingHead}, compile=True
    )  # Use custom objects to deserialize DuelinHead custom layer
    agent.set_model(model)  # type: ignore
    policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()  # type: ignore

# Create Atari frame preprocessor
from dqn.atari_utils import MultiEnvAtariFrameStacker

num_envs = 16  # 🚀 Run 16 parallel environments
frame_stacker = MultiEnvAtariFrameStacker(num_envs)

# %%
# 💪 Training using vectorized environments (faster ⚡)
envs = gym.make_vec("ALE/Breakout-v5", num_envs=num_envs, vectorization_mode="sync")

from dqn.utils.formatting import format_time

max_train_steps = int(
    1e6 * 1e3
)  # Max number of training steps (1M episodes of max 1k steps)
max_score = 500  # max score to stop training

train_t0 = None

frames, _ = envs.reset()
states = frame_stacker.reset(frames)
scores, prev_lifes = np.zeros(num_envs), np.zeros(num_envs)
for step in range(1, max_train_steps + 1):
    actions = agent.act_on_batch(states)

    frames, rewards, dones, truncs, infos = envs.step(actions)

    frame_stacker.reset_done_envs(frames, dones)
    next_states = frame_stacker.add_frames(frames)

    life_lost = dones | (prev_lifes > infos["lives"])
    clipped_rewards = np.clip(rewards, -1.0, +1.0)  # Clip reward to [-1, +1] range
    clipped_rewards[life_lost] = -1.0  # Apply life lost penalty

    batch = ExperiencesBatch(states, actions, next_states, clipped_rewards, dones)
    agent.add_experiences_batch(batch)

    states = next_states
    scores += rewards
    scores[dones] = 0.0  # Reset scores for terminated agents/environments
    prev_lifes = infos["lives"]

    metrics = None
    if agent.memory.size > 10_000:
        metrics = agent.train()

    if train_t0 is None and agent.train_steps > 0:
        train_t0 = time.time()

    loss = metrics["loss"] if metrics else np.nan

    train_elapsed = time.time() - train_t0 if train_t0 is not None else 0.0
    train_speed = agent.train_steps / train_elapsed if train_elapsed > 0.0 else 0.0

    print(
        f"Steps: {step}, "
        f"Train steps: {agent.train_steps}, "
        f"Train time: {format_time(train_elapsed)}, "
        f"Train speed: {train_speed:.2f} sps, "
        f"Memory size: {agent.memory.size}, "
        f"Max score: {scores.max()}, "
        f"Epsilon: {policy.epsilon:.4f}, "
        f"Loss: {loss:.4e}"
    )

    # Save the model each 1000 train steps
    if agent.train_steps > 0 and agent.train_steps % 1000 == 0:
        agent.model.save(filepath=model_path)
        print(f"💾 Model saved to '{model_path}'.")

    # Terminate if max episodes or max score is reached
    if agent.train_steps > max_train_steps or scores.max() > max_score:
        break

print("✅ Training completed.")
envs.close()

# Save the model
agent.model.save(filepath=model_path)
print(f"💾 Model saved to '{model_path}'.")

# %%
# 🧪 Test the trained agent

env = gym.make("ALE/Breakout-v5", render_mode="human")
frame, _ = env.reset()

from dqn.atari_utils import AtariFrameStacker

preprocessor = AtariFrameStacker()
state = preprocessor.reset(frame)

# Set exploration to zero for evaluation
policy.decay_type = "fixed"
policy.epsilon = 0.0

terminated, score, steps = False, 0, 0
while not terminated:
    env.render()

    action = agent.act(state)
    frame, reward, done, trunc, info = env.step(action)
    state = preprocessor.add_frame(frame)

    steps += 1
    score += float(reward)
    terminated = done or trunc

    print(f"Steps: {steps}, Score: {score}, Lives: {info["lives"]}")

env.close()

# %% Run all cells above
