import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)
import numpy as np
from typing import cast


class AtariWrapper(gym.Wrapper):
    """
    Custom Atari wrapper for DQN:
    - Atari preprocessing (noop, frame skip, grayscale, resize, scale)
    - Frame stacking
    - Transpose frames (C,H,W -> H,W,C) for Keras/TensorFlow
    - Reward clipping to [-1, +1]
    - Optional negative reward on life loss
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        frame_stack: int = 4,
        clip_reward: bool = True,
        life_penalty: bool = True,
    ):
        self.clip_reward = clip_reward
        self.life_penalty = life_penalty

        self.lives = 0
        self.score = 0.0

        # Apply gymansium wrappers
        env = AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=(84, 84),
            grayscale_obs=True,
        )
        env = FrameStackObservation(env, stack_size=frame_stack)

        # Replace self.env with the fully wrapped env
        super().__init__(env)
        self.observation_space._shape = (84, 84, 4)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = np.transpose(obs, axes=[1, 2, 0])

        self.score = 0.0
        info["score"] = self.score

        if self.life_penalty:
            self.lives = info.get("lives")

        return obs, info

    def step(self, action: int):
        obs, reward, done, trunc, info = self.env.step(action)
        reward = cast(float, reward)

        # Transpose observation from (S, H, W) to (H, W, S)
        obs = np.transpose(obs, axes=[1, 2, 0])

        self.score += reward
        info["score"] = self.score

        if self.life_penalty:
            current_lives = info.get("lives")
            if current_lives is None:
                raise RuntimeError("No lives in info dict.")
            if current_lives < self.lives:
                reward -= 1.0
            self.lives = current_lives

        if self.clip_reward:
            reward = np.clip(reward, a_min=-1, a_max=+1)

        return obs, reward, done, trunc, info
