# %% [markdown]
# # 🎮 DQN Agent for Atari Breakout
# A Deep Q-Network agent implemented with TensorFlow and Keras, trained to play Breakout using frame stacking and experience replay.

# %%
# 📦 Imports and Environment Setup
import os
import numpy as np
import keras.api as kr
import gymnasium as gym
import ale_py

# Ensure ALE environments are registered
gym.register_envs(ale_py)

# %%
# Test enviroment with random agent
env = gym.make("ALE/Breakout-v5", render_mode="human")
env.reset()

step, score, terminated = 0, 0, False
while not terminated:
    env.render()
    action = env.action_space.sample()
    _, reward, done, trunc, info = env.step(action)

    step += 1
    score += reward
    terminated = done or trunc

    print(f"Step: {step}, Score: {score}, Lives: {info["lives"]}")

env.close()


# %%
# 🧠 Define the DQN Model


def create_model(state_shape: tuple, num_actions: int) -> kr.Model:
    model = kr.models.Sequential(
        [
            kr.layers.InputLayer(shape=state_shape, dtype="uint8"),
            kr.layers.Rescaling(1.0 / 255.0),
            kr.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu"),
            kr.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
            kr.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
            kr.layers.Flatten(),
            kr.layers.Dense(512, activation="relu"),
            kr.layers.Dense(num_actions, activation="linear"),
        ]
    )

    model.compile(
        optimizer=kr.optimizers.Adam(learning_rate=0.00025),
        loss=kr.losses.Huber(delta=1.0),
        metrics=["accuracy"],
    )

    return model


# %%
# 🤖 Initialize the DQN Agent
from dqn import DQNAgent, EpsilonGreedyPolicy, Experience

env = gym.make("ALE/Breakout-v5")
state_shape = (84, 84, 4)
num_actions = env.action_space.n

model = create_model(state_shape, num_actions)
file_path = os.path.join("saved_models", "dqn-model-breakout.keras")
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-5)
agent = DQNAgent(
    model,
    batch_size=32,
    gamma=0.99,
    memory_size=200_000,
    policy=policy,
    update_steps=10_000,
    autosave_steps=1000,
    file_name=file_path,
    verbose=True,
)

# Load pre-trained model if it exists
if os.path.exists(file_path):
    agent.load_model(file_path, compile=True)
    agent.policy.epsilon = 0.2  # Resume with less exploration

agent.model.summary()

# Create Atari frame preprocessor
from dqn.atari_utils import AtariPreprocessor

frame_preprocessor = AtariPreprocessor()

# %%
# 🚀 Train the Agent

num_episodes = 1_000_000  # Max number of training episodes
max_score = 400  # max score to stop training

for episode in range(num_episodes):
    frame, _ = env.reset()
    state = frame_preprocessor.reset(frame)

    step, score, terminated = 0, 0, False
    while not terminated:
        action = agent.act(state)
        frame, reward, done, trunc, info = env.step(action)
        next_state = frame_preprocessor.preprocess(frame)
        clipped_reward = np.clip(reward, -1.0, +1.0)

        experience = Experience(state, action, next_state, clipped_reward, done)
        agent.add_experience(experience)
        state = next_state

        step += 1
        score += reward
        terminated = done or trunc

        if agent.memory.size > 1000 and step % 8 == 0:
            metrics = agent.train()

        if agent.train_steps > 0:
            print(
                f"Train steps: {agent.train_steps}, "
                f"Memory size: {agent.memory.size}, "
                f"Epsilon: {agent.policy.epsilon:.4f}, "
                f"Loss: {metrics["loss"]:.4e}, "
                f"Accuracy: {metrics["accuracy"]*100:.2f} %"
            )
        else:
            print(f"Episode: {episode+1}, Steps: {step}, Score: {score}")

    print(f"➡️  Episode: {episode+1}, Steps: {step}, Score: {score}")

    if score >= max_score:
        print("Max score reached.")
        break

print("✅ Training completed.")
env.close()

# %%
# 🧪 Test the trained agent

env = gym.make("ALE/Breakout-v5", render_mode="human")
frame, _ = env.reset()

from dqn.atari_utils import AtariPreprocessor

preprocessor = AtariPreprocessor()
state = preprocessor.reset(frame)

# Set exploration to zero for evaluation
agent.policy.decay_type = "fixed"
agent.policy.epsilon = 0.0

terminated, score, steps = False, 0, 0
while not terminated:
    env.render()

    action = agent.act(state)
    frame, reward, done, trunc, info = env.step(action)
    state = preprocessor.preprocess(frame)

    steps += 1
    score += reward
    terminated = done or trunc

    print(
        f"Steps: {steps}, Action: {action}, Reward: {reward}, Score: {score}, Lives: {info["lives"]}"
    )

env.close()

# %%
