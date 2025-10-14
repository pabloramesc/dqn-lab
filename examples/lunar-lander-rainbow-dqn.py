# %%
# 🌑 Rainbow DQN for LunarLander-v3
import os
import gymnasium as gym
import keras

LOAD_MODEL = True
MODEL_PATH = "models/lunar-lander-rainbow-dqn-v3.keras"


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
from keras.models import Model
from keras.layers import Dense, Input
from keras.losses import Huber
from keras.optimizers import Adam
from dqn.layers import DuelingHead, NoisyDense


def create_model(state_shape: tuple[int, ...], num_actions: int) -> Model:
    inputs = Input(shape=state_shape, dtype="float32")
    x = Dense(64, activation="relu")(inputs)

    # Value head
    v = NoisyDense(64, activation="relu")(x)
    v = NoisyDense(1, activation="linear")(v)

    # Advantage head
    a = NoisyDense(64, activation="relu")(x)
    a = NoisyDense(num_actions, activation="linear")(a)

    # Combine value and advantage: Q(s, a) = V(s) + A(s, a)
    q = DuelingHead(dtype="float32")([v, a])

    model = Model(inputs=inputs, outputs=q)
    model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=1e-3))  # type: ignore
    return model


# Initialize state and action space dimensions
env = gym.make("LunarLander-v3")
state_shape = env.observation_space.shape
num_actions = env.action_space.n  # type: ignore

# Create or load the model
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    model = keras.models.load_model(
        MODEL_PATH, compile=True, custom_objects={"DuelingHead": DuelingHead}
    )
    print(f"➡️  Loaded model from '{MODEL_PATH}'.")

else:
    model = create_model(state_shape, num_actions)  # type: ignore
    print("➡️  New model created.")

model.summary()  # type: ignore

# Create the DQN agent
from dqn import RainbowDQN, EpsilonGreedyPolicy

policy = EpsilonGreedyPolicy(
    epsilon=0.01,
    decay_type="fixed",
)

agent = RainbowDQN(
    model=model,
    policy=policy,
    batch_size=32,
    memory_size=100_000,
    update_freq=1000,
    gamma=0.99,
    n_step=3,
    alpha=0.6,
    beta=0.4,
)


# %%
# 💪 Train the agent
agent.learn(env, min_memory=1000, max_episodes=1000, model_path=MODEL_PATH)  # type: ignore


# %%
# 🧪 Test the trained agent
env = gym.make("LunarLander-v3", render_mode="human")
agent.evaluate(env, episodes=5)  # type: ignore


# %% Run all cells above
