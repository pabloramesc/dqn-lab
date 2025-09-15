from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike

from .types import FloatArray, IntArray, BoolArray, FloatLike, IntLike, BoolLike


class Experience(NamedTuple):
    """A class representing a single experience in a DQN agent training."""

    state: NDArray
    action: int
    next_state: NDArray
    reward: float
    done: bool

    @classmethod
    def create(
        cls,
        state: ArrayLike,
        action: IntLike,
        next_state: ArrayLike,
        reward: FloatLike,
        done: BoolLike,
    ):
        """Factory method to enforce type conversion and check state consistency."""
        state = np.asarray(state)
        next_state = np.array(next_state)

        if state.shape != next_state.shape:
            raise ValueError("State and next state must have same shape.")

        if state.dtype != next_state.dtype:
            raise ValueError("State and next state must have same data type.")

        return cls(
            state=state,
            action=int(action),
            next_state=next_state,
            reward=float(reward),
            done=bool(done),
        )


class ExperiencesBatch:
    """A class representing a batch of experiences, typically used for training."""

    def __init__(
        self,
        states: NDArray,
        actions: IntArray,
        next_states: NDArray,
        rewards: FloatArray,
        dones: BoolArray,
        indices: Optional[IntArray] = None,
        weights: Optional[FloatArray] = None,
    ):
        self.states = np.asarray(states)
        self.actions = np.asarray(actions, dtype=np.int32)
        self.next_states = np.asarray(next_states)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.bool_)
        self.indices = (
            np.asarray(indices, dtype=np.int32) if indices is not None else None
        )
        self.weights = (
            np.asarray(weights, dtype=np.float32) if weights is not None else None
        )
        self._check_consistency()

    @property
    def size(self) -> int:
        """Number of experiences in the batch."""
        return self.states.shape[0]

    def to_experiences(self) -> list[Experience]:
        """Convert the batch to a list of `Experience` objects."""
        experiences = [
            Experience.create(
                state=self.states[i],
                action=self.actions[i],
                next_state=self.next_states[i],
                reward=self.rewards[i],
                done=self.dones[i],
            )
            for i in range(self.size)
        ]
        return experiences

    @classmethod
    def from_experiences(
        cls, experiences: list[Experience], indices: Optional[IntArray] = None
    ):
        """Create an ExperiencesBatch from a list of Experience objects."""
        states, actions, next_states, rewards, dones = zip(*experiences)
        return cls(states, actions, next_states, rewards, dones, indices)  # type: ignore

    def _check_consistency(self):
        if self.states.shape != self.next_states.shape:
            raise ValueError("States and next states must have same shape.")
        if self.states.dtype != self.next_states.dtype:
            raise ValueError("States and next states must have same dtype.")

        arrays_1d = {
            "Actions": self.actions,
            "Rewards": self.rewards,
            "Dones": self.dones,
            "Indices": self.indices,
            "Weights": self.weights,
        }

        for name, arr in arrays_1d.items():
            if arr is not None and arr.shape != (self.size,):
                raise ValueError(f"{name} must be a 1D array of size {self.size}.")
