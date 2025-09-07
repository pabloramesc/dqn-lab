import numpy as np

from .frame_processing import process_atari_frame


class AtariFrameStacker:
    """A class to process and stack frames for Atari environments."""

    def __init__(self, stack_size: int = 4) -> None:
        """Initializes the processor with a specified stack size.

        Args:
            stack_size: The number of frames to stack for the state representation.
        """
        self.stack_size = stack_size
        self.frames: list[np.ndarray] = []

    def get_stacked_frames(self) -> np.ndarray:
        """Returns the stacked frames as the current state.

        Returns:
            The stacked frames in a single array.
        """
        return np.stack(self.frames, axis=-1)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Resets the processor with the initial frame by processing it and
        stacking it multiple times.

        Args:
            frame: The initial Atari frame.

        Returns:
            The state after reset (stacked frames).
        """
        processed_frame = process_atari_frame(frame)
        self.frames = [processed_frame] * self.stack_size
        return self.get_stacked_frames()

    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a new frame, adds it to the frame stack,
        and returns the updated state.

        Args:
            frame: The new Atari frame to be processed.

        Returns:
            The updated state after adding the new frame to the stack.
        """
        if len(self.frames) != self.stack_size:
            self.reset(frame)
            return self.get_stacked_frames()
        processed_frame = process_atari_frame(frame)
        self.frames.pop(0)
        self.frames.append(processed_frame)
        return self.get_stacked_frames()
