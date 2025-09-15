# %%
# 🕹️ DQN for Atari Breakout with VGG-style CNN, dueling Q-network,
# prioritized experience replay (PER), and vectorized environments.


# %%
# 📦 Import modules and setup environment
import os
from typing import cast

import ale_py
import gymnasium as gym
import keras
import numpy as np

# Ensure ALE environments are registered
gym.register_envs(ale_py)

# %%
# 🧠 Dueling DQN with VGG-style convolutional layers
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Rescaling
from keras.losses import Huber
from keras.models import Model
from keras.optimizers import Adam

from dqn.utils import DuelingHead


def create_model(state_shape: tuple, num_actions: int) -> Model:
    inputs = Input(shape=state_shape, dtype="uint8")

    # Normalize inputs
    x = Rescaling(1.0 / 255.0)(inputs)

    # VGG-style convolutional layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    # Dueling DQN streams
    value = Dense(1)(x)
    advantage = Dense(num_actions)(x)

    # Combine value and advantage
    q_values = DuelingHead()([value, advantage])

    model = Model(inputs=inputs, outputs=q_values)
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
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-5)
agent = DQNAgentPER(
    model=model,
    policy=policy,
    batch_size=64,
    memory_size=100_000,
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
    policy.epsilon = 1.0  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()  # type: ignore

# Create Atari frame preprocessor
from dqn.atari_utils import MultiEnvAtariFrameStacker

num_envs = 16  # 🚀 Run 16 parallel environments
frame_stacker = MultiEnvAtariFrameStacker(num_envs)

# %%
# 💪 Training using vectorized environments (faster ⚡)
envs = gym.make_vec("ALE/Breakout-v5", num_envs=num_envs, vectorization_mode="sync")

max_train_steps = int(
    1e6 * 1e3
)  # Max number of training steps (1M episodes of max 1k steps)
max_score = 500  # max score to stop training

frames, _ = envs.reset()
states = frame_stacker.reset(frames)
scores, prev_lives = np.zeros(num_envs), np.zeros(num_envs)
for step in range(1, max_train_steps + 1):
    actions = agent.act_on_batch(states)

    frames, rewards, dones, truncs, infos = envs.step(actions)

    frame_stacker.reset_done_envs(frames, dones)
    next_states = frame_stacker.add_frames(frames)

    live_lost = dones | (prev_lives > infos["lives"])
    clipped_rewards = np.where(live_lost, -1.0, np.clip(rewards, -1.0, +1.0))

    batch = ExperiencesBatch(states, actions, next_states, clipped_rewards, dones)
    agent.add_experiences_batch(batch)

    states = next_states
    scores = (scores + rewards) * (~dones)
    prev_lives = infos["lives"]

    metrics = None
    if agent.memory.size > 10_000:
        metrics = agent.train()

    loss = metrics["loss"] if metrics else np.nan

    print(
        f"Steps: {step}, "
        f"Train steps: {agent.train_steps}, "
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
