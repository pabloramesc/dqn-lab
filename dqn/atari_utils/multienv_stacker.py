import numpy as np
from numpy.typing import ArrayLike

from .frame_processing import process_atari_frame
from .frame_stacker import AtariFrameStacker


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
        self.stackers: list[AtariFrameStacker] = []
        for _ in range(self.num_envs):
            self.stackers.append(AtariFrameStacker(self.stack_size))

    def get_stacked_frames(self) -> np.ndarray:
        """
        Returns the current states of all environments as a stacked array.

        Returns
        -------
        np.ndarray
            The stacked states for all environments.
        """
        states = np.stack([s.get_stacked_frames() for s in self.stackers], axis=0)
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
        states = []
        for i, stacker in enumerate(self.stackers):
            state = stacker.reset(frames[i])
            states.append(state)
        return np.stack(states, axis=0)

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
            p: AtariFrameStacker = self.stackers[i]
            p.reset(frames[i])

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
        states = []
        for i, stacker in enumerate(self.stackers):
            state = stacker.add_frame(frames[i])
            states.append(state)
        return np.stack(states, axis=0)
