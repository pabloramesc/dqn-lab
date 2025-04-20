# %%
# 🚕 Q-Learning for Taxi-v3
# A simple implementation using NumPy and OpenAI Gym

import numpy as np
import gym

# %%
# 📦 Environment and Hyperparameters

env = gym.make("Taxi-v3")

alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.9995  # Exploration decay per episode
min_epsilon = 0.01  # Minimum exploration rate

num_episodes = 10_000  # Max number of training episodes
max_steps = 100  # Max number of steps per episode

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# %%
# 🎮 Helper functions


def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explore (random action)
    return np.argmax(q_table[state])  # Exploit (best Q-table action)


def update_q_table(state, action, reward, next_state):
    best_next = np.max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
        reward + gamma * best_next
    )


# %%
# 🧠 Training loop

for episode in range(num_episodes):
    state, _ = env.reset()

    score = 0
    for step in range(max_steps):
        action = choose_action(state, epsilon)
        new_state, reward, done, truncated, info = env.step(action)

        update_q_table(state, action, reward, new_state)

        state = new_state
        score += reward

        if done or truncated:
            break

    print(f"Episode: {episode+1}, Steps: {step+1}, Score: {score}, Epsilon: {epsilon}")

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()


# %%
# 🧪 Test the trained agent

env = gym.make("Taxi-v3", render_mode="human")

for episode in range(5):
    state, _ = env.reset()

    score = 0
    for step in range(max_steps):
        env.render()

        action = choose_action(state, epsilon=0.0)  # Only exploit during testing
        new_state, reward, done, truncated, info = env.step(action)

        state = new_state
        score += reward

        if done or truncated:
            print(
                f"Episode: {episode+1} finished after {step+1} steps with score {score}"
            )
            break

env.close()

# %%
