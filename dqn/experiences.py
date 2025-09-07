import numpy as np
from numpy.typing import NDArray

class Experience:
    """A class representing a single experience in a DQN agent training."""

    def __init__(
        self,
        state: NDArray,
        action: int,
        next_state: NDArray,
        reward: float,
        done: bool,
    ):
        self.state = np.asarray(state)
        self.next_state = np.asarray(next_state)
        self.action = int(action)
        self.reward = float(reward)
        self.done = bool(done)
        self._check_consistency()

    def to_tuple(self) -> tuple[NDArray, int, NDArray, float, bool]:
        """Convert the experience into a tuple format."""
        return (self.state, self.action, self.next_state, self.reward, self.done)

    def _check_consistency(self):
        if self.state.shape != self.next_state.shape:
            raise ValueError("State and next state must have same shape.")
        if self.state.dtype != self.next_state.dtype:
            raise ValueError("State and next state must have same data type.")


class ExperiencesBatch:
    """A class representing a batch of experiences, typically used for training."""

    def __init__(
        self,
        states: NDArray,
        actions: NDArray[np.int32],
        next_states: NDArray,
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        indices: NDArray[np.int32] | None = None,
        weights: NDArray[np.float32] | None = None,
    ):
        self.states = np.asarray(states)
        self.next_states = np.asarray(next_states)
        self.actions = np.asarray(actions, dtype=np.int32)
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
            Experience(
                state=self.states[i],
                action=self.actions[i],
                next_state=self.next_states[i],
                reward=self.rewards[i],
                done=self.dones[i],
            )
            for i in range(self.size)
        ]
        return experiences

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
