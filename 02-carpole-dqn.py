# %%
# 🏃‍♂️ DQN for CartPole-v1
# A simple implementation using TensorFlow/Keras and OpenAI Gym

import gym
import keras.api as kr
import os
from dqn import DQNAgent, EpsilonGreedyPolicy, Experience

# %%
# 🎮 Random Environment Test

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(5):
    state, _ = env.reset()  # Reset environment and get initial state

    score = 0
    for step in range(100):
        env.render()  # Render the environment

        # Choose random action for testing
        action = env.action_space.sample()
        new_state, reward, done, trunc, info = env.step(action)

        state = new_state
        score += reward

        if done or trunc:
            print(
                f"Episode: {episode+1} finished after {step+1} steps with score {score}"
            )
            break

env.close()


# %%
# 🎮 DQN Model Definition


def create_model(state_shape: tuple, num_actions: int) -> kr.Model:
    model = kr.models.Sequential()
    model.add(kr.layers.Dense(24, activation="relu", input_shape=state_shape))
    model.add(kr.layers.Dense(24, activation="relu"))
    model.add(kr.layers.Dense(num_actions, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


# Initialize state and action space dimensions
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape
num_actions = env.action_space.n

# Create DQN agent
file_path = os.path.join("saved_models", "dqn-model-cartpole.keras")
model = create_model(state_shape, num_actions)
policy = EpsilonGreedyPolicy(
    decay_type="exponential", epsilon_min=0.01, epsilon_decay=0.9995
)
agent = DQNAgent(
    model,
    batch_size=32,
    gamma=0.95,
    policy=policy,
    memory_size=100_000,
    update_steps=100,
    autosave_steps=100,
    file_name=file_path,
    verbose=True,
)

# Load model if it exists
if os.path.exists(file_path):
    agent.load_model(file_path, compile=True)
    agent.policy.epsilon = 0.1

agent.model.summary()


# %%
# 🧠 Training Loop

num_episodes = 10_000  # Max number of training episodes
for episode in range(num_episodes):
    state, _ = env.reset()  # Reset environment and get initial state

    step, score, terminated = 0, 0, False
    while not terminated:
        action = agent.act(state)  # Choose action based on policy
        new_state, reward, done, trunc, info = env.step(action)

        exp = Experience(state, action, new_state, reward, done)
        agent.add_experience(exp)
        state = new_state

        step += 1
        score += reward
        terminated = done or trunc

        # Train when enough experiences in buffer and each few steps
        if agent.memory.size > 100 and step % 8 == 0:
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

    print(f"➡️  Episode: {episode+1}, Steps: {step+1}, Score: {score}")

    if trunc:
        print("Game truncated at max score/steps.")
        break  # End training if truncated (max score reached)

print("✅ Training completed.")
env.close()

# %%
# 🧪 Test the trained agent
env = gym.make("CartPole-v1", render_mode="human")  # Create environment for testing

for episode in range(5):  # Test for 5 episodes
    state, _ = env.reset()  # Reset environment and get initial state

    step, score, terminated = 0, 0, False
    while not terminated:
        env.render()

        action = agent.act(state)
        new_state, reward, done, trunc, info = env.step(action)

        state = new_state
        score += reward
        terminated = done or trunc

        if terminated:
            print(
                f"Episode: {episode+1} finished after {step+1} steps with score {score}"
            )
            break

env.close()

# %%
