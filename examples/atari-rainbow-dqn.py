# %%
# 🕹️ Rainbow DQN Agent for Atari games
LOAD_MODEL = True
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
MODEL_PATH = "models/space-invaders-rainbow-dqn.keras"

print("🕹️  Game environment:", GAME_NAME)


# %%
# 📦 Import modules and setup environment
import os
import ale_py
import gymnasium as gym

# Ensure ALE environments are registered
gym.register_envs(ale_py)


# %%
# 🤖 Initialize the DQN agent
from dqn.wrappers import AtariWrapper
from dqn.models import build_atari_rainbow_dqn, load_atari_rainbow_dqn
from gymnasium.vector import AsyncVectorEnv


# Create vectorized environments
def make_env():
    env = gym.make(GAME_NAME)
    env = AtariWrapper(env)
    return env


num_envs = 8  # 🚀 Run parallel environments
envs = AsyncVectorEnv([make_env for _ in range(num_envs)])

state_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n  # type: ignore

# Create the policy model
model = build_atari_rainbow_dqn(state_shape, num_actions)  # type: ignore

# Load pre-trained model if it exists
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model = load_atari_rainbow_dqn(MODEL_PATH)
    agent.set_model(model)  # type: ignore
    print(f"➡️  Model loaded from '{MODEL_PATH}'.")

model.summary()  # type: ignore


from dqn import RainbowDQN, EpsilonGreedyPolicy

policy = EpsilonGreedyPolicy(decay_type="fixed", epsilon=0.01)
agent = RainbowDQN(
    model=model,
    batch_size=32,
    memory_size=200_000,  # aprox. 2.6 GB of RAM per 100k samples
    policy=policy,
    update_freq=10_000,
    gamma=0.99,
    n_step=3,
    alpha=0.6,
    beta=0.4,
    beta_annealing=1e-6,
)


# %%
# 💪 Train the agent
agent.learn_parallel(
    envs=envs,  # type: ignore
    max_episodes=1_000_000,
    min_memory=50_000,
    train_every=4,
    model_path=MODEL_PATH,
    autosave_freq=10_000,
    verbose=2,
)


# %%
# 🧪 Test the trained agent
env = gym.make(GAME_NAME, render_mode="human")
env = AtariWrapper(env)

agent.evaluate(env)  # type: ignore


# %% Run all cells above
