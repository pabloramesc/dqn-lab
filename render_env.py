import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # Ensure ALE environments are registered

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

for episode in range(5):
    state, _ = env.reset()  # Reset environment and get initial state

    score = 0
    for steps in range(1000):
        env.render()  # Render the environment

        # Choose random action for testing
        action = env.action_space.sample()
        new_state, reward, done, trunc, info = env.step(action)

        state = new_state
        score += float(reward)

        if done or trunc:
            break

    print(f"Episode {episode+1} finished after {steps+1} steps with score {score}")

env.close()
