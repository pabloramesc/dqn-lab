# %%
# 🕹️ DQN Agent for Atari games
LOAD_MODEL = False
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
MODEL_PATH = "models/space-invaders-vanilla-dqn.keras"

print("🕹️  Game environment:", GAME_NAME)


# %%
# 📦 Import modules and setup environment
import os
import ale_py
import gymnasium as gym
import keras

# Ensure ALE environments are registered
gym.register_envs(ale_py)


# %%
# 🤖 Initialize the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy
from dqn.wrappers import AtariWrapper
from dqn.models import build_atari_vanilla_dqn

# Initialize environment state and action space dimensions
env = gym.make(GAME_NAME, render_mode="rgb_array")
env = AtariWrapper(env, frame_skip=4)
state_shape = env.observation_space.shape
num_actions = env.action_space.n  # type: ignore

# Create or load the model
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=True)
    print(f"➡️ Loaded model from '{MODEL_PATH}'.")
    epsilon0 = 0.1  # Less exploration if continuing from pretrained model

else:
    model = build_atari_vanilla_dqn(state_shape, num_actions)  # type: ignore
    print("➡️ New model created.")
    epsilon0 = 1.0  # Start with full exploration for fresh training

model.summary()  # type: ignore

# Create the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy

policy = EpsilonGreedyPolicy(
    epsilon=epsilon0,
    epsilon_min=0.01,
    decay_type="linear",
    epsilon_decay=1e-4,
)

agent = DQNAgent(
    model=model,
    policy=policy,
    batch_size=32,
    memory_size=100_000,
    gamma=0.99,
    update_freq=1000,
)


# %%
# 💪 Train the agent
agent.learn(
    env=env,  # type: ignore
    max_episodes=10_000,
    min_memory=50_000,
    train_every=4,
    model_path=MODEL_PATH,
    autosave_freq=10_000,
    verbose=2,
)


# %%
# 🧪 Test the trained agent
env = gym.make(GAME_NAME, render_mode="human")
env = AtariWrapper(env, frame_skip=4)

agent.evaluate(env)  # type: ignore


# %% Run all cells above
