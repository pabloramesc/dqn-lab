# %%
# 🌑 DQN for LunarLander-v3
import os
import gymnasium as gym
import keras

LOAD_MODEL = True
MODEL_PATH = "models/lunar-lander-vanilla-dqn.keras"


# %%
# 🎮 Environment demo with random actions

env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(5):
    obs, info = env.reset()  # Reset environment and get initial state

    score = 0.0
    for steps in range(1000):
        action = env.action_space.sample()  # Choose a random action
        obs, reward, done, trunc, info = env.step(action)
        score += float(reward)

        print(
            f"Episode: {episode+1}, steps: {steps+1}, score: {score:.1f}",
            end="        \r",
        )

        if done or trunc:
            break

    print()

env.close()


# %%
# 🧠 DQN model and agent definition
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.losses import Huber
from keras.optimizers import Adam


def create_model(state_shape: tuple[int, ...], num_actions: int) -> Model:
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=state_shape))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_actions, activation="linear"))
    model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=1e-3))  # type: ignore
    return model


# Initialize state and action space dimensions
env = gym.make("LunarLander-v3")
state_shape = env.observation_space.shape
num_actions = env.action_space.n  # type: ignore

# Create or load the model
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=True)
    print(f"➡️ Loaded model from '{MODEL_PATH}'.")
    epsilon0 = 0.1  # Less exploration if continuing from pretrained model

else:
    model = create_model(state_shape, num_actions)  # type: ignore
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
agent.learn(env, min_memory=1000, max_episodes=1000, model_path=MODEL_PATH)  # type: ignore


# %%
# 🧪 Test the trained agent
env = gym.make("LunarLander-v3", render_mode="human")
agent.evaluate(env, episodes=5)  # type: ignore


# %% Run all cells above
