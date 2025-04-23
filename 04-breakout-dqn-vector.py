# %% [markdown]
# # 🎮 DQN for Atari Breakout - Vectorized Version 🧠🚀
# We train a DQN agent to play the classic Atari *Breakout*, using **vectorized environments** with `gymnasium` to collect experiences in parallel and **speed up training**.


# %%
# ## 📦 Imports & Environment registration
import os
import numpy as np
import keras.api as kr
import gymnasium as gym
import ale_py

# Ensure ALE environments are registered
gym.register_envs(ale_py)

# %%
# 🧠 DQN model with CNN (DeepMind style)


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
# 🤖 Configure the DQN agent + prioritized replay buffer
from dqn import DQNAgent, EpsilonGreedyPolicy, ExperiencesBatch, PriorityReplayBuffer

# Env settings
env = gym.make("ALE/Breakout-v5")
state_shape = (84, 84, 4)
num_actions = env.action_space.n

# Create the model and agent
model = create_model(state_shape, num_actions)
file_path = file_path = os.path.join("saved_models", "dqn-model-breakout.keras")
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-6)
memory = PriorityReplayBuffer(
    max_size=250_000, alpha=0.6, beta=0.4, beta_annealing=0.0
)
agent = DQNAgent(
    model,
    batch_size=64,
    gamma=0.99,
    policy=policy,
    # memory_size=200_000, <-- not using default buffer
    memory=memory,
    update_steps=10_000,
    autosave_steps=1000,
    file_name=file_path,
    verbose=True,
)

# Load pre-trained model if it exists
if os.path.exists(file_path):
    agent.load_model(file_path, compile=True)
    agent.policy.epsilon = 1.0  # Resume with less exploration

agent.model.summary()


# Create Atari frame preprocessor
from dqn.atari_utils import MultiEnvAtariFrameStacker

num_envs = 16  # 🚀 Run 16 parallel environments
frame_stacker = MultiEnvAtariFrameStacker(num_envs)

# %%
# 🎓 Training using vectorized envs (faster ⚡)
envs = gym.make_vec("ALE/Breakout-v5", num_envs=num_envs, vectorization_mode="sync")

num_episodes = 100_000_000  # Max number of training episodes
max_score = 400  # max score to stop training

frames, _ = envs.reset()
states = frame_stacker.reset(frames)
scores = np.zeros(num_envs)
while True:
    actions = agent.act_on_batch(states)

    frames, rewards, dones, truncs, infos = envs.step(actions)
    next_states = frame_stacker.add_frames(frames)

    clipped_rewards = np.clip(rewards, -1.0, +1.0)
    batch = ExperiencesBatch(states, actions, next_states, clipped_rewards, dones)
    agent.add_experiences_batch(batch)

    states = next_states
    scores = (scores + rewards) * (~dones)
    
    frame_stacker.reset_done_envs(frames, dones)

    if agent.memory.size > 50_000:
        metrics = agent.train()

    if agent.train_steps > 0:
        print(
            f"Train steps: {agent.train_steps}, "
            f"Memory size: {agent.memory.size}, "
            f"Max score: {scores.max()}, "
            f"Epsilon: {agent.policy.epsilon:.4f}, "
            f"Loss: {metrics["loss"]:.4e}, "
            f"Accuracy: {metrics["accuracy"]*100:.2f} %"
        )
    else:
        print(
            f"Memory size: {agent.memory.size}, "
            f"Max score: {scores.max()}, "
            f"Epsilon: {agent.policy.epsilon:.4f}, "
        )

    if agent.train_steps > num_episodes or scores.max() > max_score:
        break

print("✅ Training completed.")
envs.close()

# %%
# 🧪 Test the trained agent

env = gym.make("ALE/Breakout-v5", render_mode="human")
frame, _ = env.reset()

from dqn.atari_utils import AtariFrameStacker

preprocessor = AtariFrameStacker()
state = preprocessor.reset_stack(frame)

# Set exploration to zero for evaluation
agent.policy.decay_type = "fixed"
agent.policy.epsilon = 0.0

terminated, score, steps = False, 0, 0
while not terminated:
    env.render()

    action = agent.act(state)
    frame, reward, done, trunc, info = env.step(action)
    state = preprocessor.add_frame(frame)

    steps += 1
    score += reward
    terminated = done or trunc

    print(f"Steps: {steps}, Score: {score}, Lives: {info["lives"]}")

env.close()

# %%
