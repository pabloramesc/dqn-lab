import cv2
import numpy as np
from matplotlib import pyplot as plt

from .frame_stacker import AtariFrameStacker

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

    gym.register_envs(ale_py)

    env = gym.make("ALE/Breakout-v5")
    rgb_state = env.reset()[0]

    # process frame function
    gray_state = cv2.cvtColor(rgb_state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 110))
    cropped_state = resized_state[18:102, :]
    normalized_state = (cropped_state / 255.0).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    images = [
        (rgb_state, "RGB State (210x160x3)"),
        (gray_state, "Gray State (210x160x1)"),
        (resized_state, "Resized State (110x84x1)"),
        (cropped_state, "Cropped State (84x84x1)"),
    ]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    frame_processor = AtariFrameStacker()
    frame = env.reset()[0]
    state = frame_processor.reset(frame)
    for i in range(100):
        print(f"Step {i}")
        action = env.action_space.sample()
        frame, reward, done, trunc, info = env.step(action)
        state = frame_processor.add_frame(frame)
        if i >= 30:
            plot_stacked_frames(state)

    env.close()
