import cv2
import numpy as np


def normalize_frame(uint8_frame: np.ndarray) -> np.ndarray:
    """Normalizes an 8-bit uint8 frame (0-255) to a floating point frame with
    values in the range [0, 1].

    Args:
        uint8_frame: The input image frame with uint8 type.

    Returns:
        The normalized image frame with float32 type.
    """
    if uint8_frame.dtype != np.uint8:
        return uint8_frame
    normalized_frame = (uint8_frame / 255.0).astype(np.float32)
    return normalized_frame


def process_atari_frame(rgb_frame: np.ndarray) -> np.ndarray:
    """Processes an Atari game frame for neural network input.
    
    This includes converting to grayscale, resizing the image, and cropping it.

    The original Atari frame size is (210, 160, 3) with RGB channels.
    After processing, the frame has a size of (84, 84, 1).

    Args:
        rgb_frame: The input RGB Atari frame with shape (210, 160, 3).

    Returns
        The processed grayscale frame with shape (84, 84, 1).
    """
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 110))
    cropped_frame = resized_frame[18:102, :]
    return cropped_frame