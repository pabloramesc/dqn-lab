"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike


def normalize_frame(uint8_frame: np.ndarray) -> np.ndarray:
    """
    Normalizes an 8-bit uint8 frame (0-255) to a floating point frame with
    values in the range [0, 1].

    Parameters
    ----------
    uint8_frame : np.ndarray
        The input image frame with uint8 type.

    Returns
    -------
    np.ndarray
        The normalized image frame with float32 type.
    """
    if uint8_frame.dtype != np.uint8:
        return uint8_frame
    normalized_frame = (uint8_frame / 255.0).astype(np.float32)
    return normalized_frame


def process_atari_frame(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Processes an Atari game frame for neural network input. This includes
    converting to grayscale, resizing the image, and cropping it.

    The original Atari frame size is (210, 160, 3) with RGB channels.
    After processing, the frame has a size of (84, 84, 1).

    Parameters
    ----------
    rgb_frame : np.ndarray
        The input RGB Atari frame with shape (210, 160, 3).

    Returns
    -------
    np.ndarray
        The processed grayscale frame with shape (84, 84, 1).
    """
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 110))
    cropped_frame = resized_frame[18:102, :]
    return cropped_frame


class AtariFrameStacker:
    """
    A class to process and stack frames for Atari environments.
    """

    def __init__(self, stack_size=4) -> None:
        """
        Initializes the processor with a specified stack size.

        Parameters
        ----------
        stack_size : int
            The number of frames to stack for the state representation.
        """
        self.stack_size = stack_size
        self.frames = None

    def get_stacked_frames(self) -> np.ndarray:
        """
        Returns the stacked frames as the current state.

        Returns
        -------
        np.ndarray
            The stacked frames in a single array.
        """
        return np.stack(self.frames, axis=-1)

    def reset_stack(self, frame: np.ndarray) -> np.ndarray:
        """
        Resets the processor with the initial frame by processing it
        and stacking it multiple times.

        Parameters
        ----------
        frame : np.ndarray
            The initial Atari frame.

        Returns
        -------
        np.ndarray
            The state after reset (stacked frames).
        """
        processed_frame = process_atari_frame(frame)
        self.frames = [processed_frame] * self.stack_size
        return self.get_stacked_frames()

    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a new frame, adds it to the frame stack, and returns
        the updated state.

        Parameters
        ----------
        frame : np.ndarray
            The new Atari frame to be processed.

        Returns
        -------
        np.ndarray
            The updated state after adding the new frame to the stack.
        """
        if len(self.frames) != self.stack_size:
            self.reset_stack(frame)
            return self.get_stacked_frames()
        processed_frame = process_atari_frame(frame)
        self.frames.pop(0)
        self.frames.append(processed_frame)
        return self.get_stacked_frames()


class MultiEnvAtariFrameStacker:
    """
    A class to process frames for multiple Atari environments at once.
    """

    def __init__(self, num_envs: int, stack_size=4) -> None:
        """
        Initializes the vectorized processor for multiple environments.

        Parameters
        ----------
        num_envs : int
            The number of environments to process.
        stack_size : int
            The number of frames to stack for each environment.
        """
        self.num_envs = num_envs
        self.stack_size = stack_size
        self.processors: list[AtariFrameStacker] = []
        for _ in range(self.num_envs):
            self.processors.append(AtariFrameStacker(self.stack_size))

    def get_stacked_frames(self) -> np.ndarray:
        """
        Returns the current states of all environments as a stacked array.

        Returns
        -------
        np.ndarray
            The stacked states for all environments.
        """
        states = np.stack([p.get_stacked_frames() for p in self.processors], axis=-1)
        return states

    def reset(self, frames: ArrayLike) -> np.ndarray:
        """
        Resets all environments with their respective initial frames.

        Parameters
        ----------
        frames : ArrayLike
            A batch of initial Atari frames for all environments.

        Returns
        -------
        np.ndarray
            The states of all environments after reset.
        """
        states = np.stack(
            [p.reset(frame) for p, frame in zip(self.processors, frames)], axis=0
        )
        return states

    def reset_done_envs(self, frames: ArrayLike, dones: ArrayLike) -> None:
        """
        Resets the frame stacks for environments that are marked as "done."

        Parameters
        ----------
        frames : ArrayLike
            A batch of Atari frames for all environments. Each frame
            corresponds to an environment.
        dones : ArrayLike
            A boolean array indicating which environments are "done." True
            means the environment is done and needs to be reset.
        """
        for i in np.arange(self.num_envs)[dones]:
            p: AtariFrameStacker = self.processors[i]
            p.reset_stack(frames[i])

    def add_frames(self, frames: ArrayLike) -> np.ndarray:
        """
        Processes new frames for all environments and updates their state.

        Parameters
        ----------
        frames : ArrayLike
            A batch of Atari frames for all environments.

        Returns
        -------
        np.ndarray
            The updated states for all environments.
        """
        states = np.stack(
            [p.process(frame) for p, frame in zip(self.processors, frames)],
            axis=0,
        )
        return states


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
    import gymnasium as gym
    import ale_py

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
    state = frame_processor.reset_stack(frame)
    for i in range(100):
        print(f"Step {i}")
        action = env.action_space.sample()
        frame, reward, done, trunc, info = env.step(action)
        state = frame_processor.add_frame(frame)
        if i >= 30:
            plot_stacked_frames(state)

    env.close()
