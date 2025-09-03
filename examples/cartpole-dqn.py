# %%
# 🛒 DQN for CartPole-v1

import os

import gymnasium as gym
import keras


# %%
# 🎮 Environment demo with random actions

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(5):
    state, _ = env.reset()  # Reset environment and get initial state

    score = 0
    for steps in range(100):
        env.render()  # Render the environment

        # Choose random action for testing
        action = env.action_space.sample()
        new_state, reward, done, trunc, info = env.step(action)

        state = new_state
        score += reward

        if done or trunc:
            print(
                f"Episode: {episode+1} finished after {steps+1} steps with score {score}"
            )
            break

env.close()


# %%
# 🧠 DQN model and agent definition
from keras.models import Model, Sequential
from keras.layers import Dense


def create_model(state_shape: tuple, num_actions: int) -> Model:
    model = Sequential()
    model.add(Dense(24, activation="relu", input_shape=state_shape))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(num_actions, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    return model


# Initialize state and action space dimensions
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape
num_actions = env.action_space.n

# Create the DQN agent
from dqn import DQNAgent, EpsilonGreedyPolicy, Experience

model = create_model(state_shape, num_actions)
policy = EpsilonGreedyPolicy(
    decay_type="exponential", epsilon_min=0.01, epsilon_decay=0.9995
)
agent = DQNAgent(
    model=model,
    batch_size=32,
    memory_size=100_000,
    gamma=0.95,
    policy=policy,
    update_freq=1000,
)

# Load pre-trained model if it exists
model_path = "models/cartpole-model.keras"

if os.path.exists(model_path):
    model = keras.models.load_model(filepath=model_path, compile=True)
    agent.set_model(model)
    agent.policy.epsilon = 0.1  # Resume with less exploration
    print(f"➡️  Model loaded from '{model_path}'.")

model.summary()


# %%
# 💪 Training loop

max_episodes = 10_000  # Max number of training episodes
for episode in range(max_episodes):
    state, _ = env.reset()  # Reset environment and get initial state

    steps, score, terminated = 0, 0, False
    while not terminated:
        action = agent.act(state)  # Choose action based on policy
        new_state, reward, done, trunc, info = env.step(action)

        exp = Experience(state, action, new_state, reward, done)
        agent.add_experience(exp)
        state = new_state

        steps += 1
        score += reward
        terminated = done or trunc

        # Train when enough experiences in buffer and each few steps
        if agent.memory.size > 1000 and steps % 8 == 0:
            metrics = agent.train()

        print(
            f"Episode: {episode+1}, Steps: {steps}, Score: {score}, "
            f"Memory size: {agent.memory.size}, "
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

    if trunc:
        print("Game truncated at max score/steps.")
        break  # End training if truncated (max score reached)

print("✅ Training completed.")
env.close()

# Save the model
agent.model.save(filepath=model_path)
print(f"💾 Model saved to '{model_path}'.")


# %%
# 🧪 Test the trained agent
env = gym.make("CartPole-v1", render_mode="human")  # Create environment for testing

# Set exploration to zero for evaluation
agent.policy.decay_type = "fixed"
agent.policy.epsilon = 0.0

for episode in range(5):  # Test for 5 episodes
    state, _ = env.reset()  # Reset environment and get initial state

    steps, score, terminated = 0, 0, False
    while not terminated:
        env.render()

        action = agent.act(state)
        new_state, reward, done, trunc, info = env.step(action)
        state = new_state

        steps += 1
        score += reward
        terminated = done or trunc

        if terminated:
            print(
                f"Episode: {episode+1} finished after {steps+1} steps with score {score}"
            )
            break

env.close()

# %% Run all cells above
