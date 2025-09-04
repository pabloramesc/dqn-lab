# %%
# 🕹️ DQN Agent for Atari Breakout in Google DeepMind style

# %%
# 📦 Import modules and setup environment
import os

import ale_py
import gymnasium as gym
import keras
import numpy as np

# Ensure ALE environments are registered
gym.register_envs(ale_py)

# %%
# 🎮 Test enviroment with random agent
env = gym.make("ALE/Breakout-v5", render_mode="human")
env.reset()

steps, score, terminated = 0, 0, False
while not terminated:
    env.render()
    action = env.action_space.sample()
    _, reward, done, trunc, info = env.step(action)

    steps += 1
    score += reward
    terminated = done or trunc

    print(f"Step: {steps}, Score: {score}, Lives: {info["lives"]}", end="\r")

print()
env.close()


# %%
# 🧠 Define the DQN model with CNN (DeepMind style)
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Rescaling
from keras.optimizers import Adam
from keras.losses import Huber


def create_model(state_shape: tuple, num_actions: int) -> Model:
    model = Sequential(
        [
            InputLayer(shape=state_shape, dtype="uint8"),
            Rescaling(1.0 / 255.0),
            Conv2D(32, (8, 8), strides=(4, 4), activation="relu"),
            Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
            Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(num_actions, activation="linear"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.00025),
        loss=Huber(delta=1.0),
    )

    return model


# %%
# 🤖 Initialize the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy, Experience

# Initialize state and action space dimensions
env = gym.make("ALE/Breakout-v5")
state_shape = (84, 84, 4)
num_actions = env.action_space.n

# Create the DQN agent
model = create_model(state_shape, num_actions)
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-5)
agent = DQNAgent(
    model=model,
    batch_size=32,
    gamma=0.99,
    memory_size=200_000,
    policy=policy,
    update_freq=1000,
)

# Load pre-trained model if it exists
model_path = "models/breakout-deepmind-model.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(filepath=model_path, compile=True)
    agent.set_model(model)
    agent.policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()

# Create Atari frame preprocessor
from dqn.atari_utils import AtariFrameStacker

frame_stacker = AtariFrameStacker()

# %%
# 💪 Train the agent

max_episodes = 1_000_000  # Max number of training episodes
max_score = 400  # Max score to stop training

for episode in range(max_episodes):
    frame, info = env.reset()
    state = frame_stacker.reset(frame)

    prev_lives = info["lives"]
    steps, score, terminated = 0, 0, False
    while not terminated:
        action = agent.act(state)
        frame, reward, done, trunc, info = env.step(action)
        next_state = frame_stacker.add_frame(frame)

        live_lost = done or info["lives"] < prev_lives
        clipped_reward = np.clip(reward, -1.0, +1.0) if not live_lost else -1.0

        experience = Experience(state, action, next_state, clipped_reward, done)
        agent.add_experience(experience)

        state = next_state
        steps += 1
        score += reward
        terminated = done or trunc
        prev_lives = info["lives"]

        if agent.memory.size > 1000 and steps % 4 == 0:
            metrics = agent.train()

        print(
            f"Episode: {episode+1}, Steps: {steps}, Score: {score}, "
            f"Lives: {info["lives"]}, Memory size: {agent.memory.size}, "
            f"Epsilon: {agent.policy.epsilon:.4f}",
            end="",
        )

        if agent.train_steps > 0:
            print(
                f", Train steps: {agent.train_steps}, Loss: {metrics["loss"]:.4e}",
                end="\r",
            )
        else:
            print(end="\r")

    print()

    # Save the model after each episode
    agent.model.save(filepath=model_path)

    if score >= max_score:
        print("Max score reached.")
        break

print("✅ Training completed.")
env.close()

# Save the model
agent.model.save(filepath=model_path)
print(f"💾 Model saved to '{model_path}'.")


# %%
# 🧪 Test the trained agent
env = gym.make("ALE/Breakout-v5", render_mode="human")
frame, _ = env.reset()

frame_stacker = AtariFrameStacker()
state = frame_stacker.reset(frame)

# Set exploration to zero for evaluation
agent.policy.decay_type = "fixed"
agent.policy.epsilon = 0.0

terminated, score, steps = False, 0, 0
while not terminated:
    env.render()

    action = agent.act(state)
    frame, reward, done, trunc, info = env.step(action)
    state = frame_stacker.add_frame(frame)

    steps += 1
    score += reward
    terminated = done or trunc

    print(
        f"Steps: {steps}, Action: {action}, Reward: {reward}, Score: {score}, Lives: {info["lives"]}"
    )

env.close()

# %% Run all cells above
