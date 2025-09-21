import numpy as np
from matplotlib import pyplot as plt

def plot_stacked_frames(state: np.ndarray):
    """
    Plots the processed frames from a stacked Atari state.

    Parameters
    ----------
    state : np.ndarray
        A state containing stacked Atari frames (84x84x4).

    Raises
    ------
    ValueError
        If the state does not have the shape (84, 84, 4).
    """
    if state.shape != (84, 84, 4):
        raise ValueError("Atari prepocessed frames must be (84, 84, 4) shaped")

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    for i, ax in enumerate(axes):
        ax.imshow(state[:, :, i])
        ax.set_title(f"Frame {i} (84x84x1)")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import ale_py
    import gymnasium as gym
    from dqn.wrappers import AtariWrapper

    gym.register_envs(ale_py)

    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariWrapper(env)
    
    env.reset()
    
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, trunc, info = env.step(action)
        print(f"Step: {i}, Action: {action}, Reward: {reward}, Done: {done}")
        plot_stacked_frames(state)

    env.close()
