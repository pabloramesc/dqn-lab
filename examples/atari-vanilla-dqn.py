# %%
# 🕹️ DQN Agent for Atari games
GAME_NAME = "SpaceInvadersNoFrameskip-v4"
MODEL_PATH = "models/space-invaders-vanilla-dqn-f32.keras"

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
from dqn import DQNAgent, EpsilonGreedyPolicy
from dqn.wrappers import AtariWrapper
from dqn.models import build_atari_vanilla_dqn

# Initialize environment state and action space dimensions
env = gym.make(GAME_NAME, render_mode="rgb_array")
env = AtariWrapper(env, frame_skip=4)

state_shape = env.observation_space.shape
num_actions = env.action_space.n  # type: ignore

# Create the DQN agent
model = build_atari_vanilla_dqn(state_shape, num_actions)  # type: ignore
policy = EpsilonGreedyPolicy(decay_type="linear", epsilon_min=0.1, epsilon_decay=1e-5)
agent = DQNAgent(
    model=model,
    batch_size=32,
    gamma=0.99,
    memory_size=200_000,  # aprox. 2.6 GB of RAM per 100k samples
    policy=policy,
    update_freq=10_000,
)

# Load pre-trained model if it exists
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(filepath=MODEL_PATH, compile=True)
    agent.set_model(model)  # type: ignore
    policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{MODEL_PATH}'.")

model.summary()  # type: ignore


# %%
# 💪 Train the agent
agent.learn(
    env=env,  # type: ignore
    max_episodes=10_000,
    min_memory=50_000,
    train_every=1,
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
