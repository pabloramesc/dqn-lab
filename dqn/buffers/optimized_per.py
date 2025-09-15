import numpy as np
from numpy.typing import NDArray

from dqn.experiences import Experience, ExperiencesBatch

from .per_buffer import PERBuffer
from .sum_tree import SumTree
from ..types import IntArray, FloatArray


class OptimizedPER(PERBuffer):

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.000001,
        min_priority: float = 0.000001,
    ):
        super().__init__(max_size, alpha, beta, beta_annealing, min_priority)
        self.sum_tree = SumTree(capacity=self.max_size)
        # Remove priorities buffer for safe
        self.ptr = None
        self.priorities = None

    def add(self, exp: Experience, td_error: float = 1) -> None:
        self.buffer.add(exp)
        priority = max(td_error, self.min_priority)
        self.sum_tree.add(priority)

    def add_batch(
        self, batch: ExperiencesBatch, td_errors: NDArray[np.float32] | None = None
    ) -> None:
        self.buffer.add_batch(batch)

        if td_errors is None:
            priorities = np.ones(batch.size, dtype=np.float32)
        else:
            priorities = np.clip(td_errors, a_min=self.min_priority, a_max=None)

        for priority in priorities:
            self.sum_tree.add(priority)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        total = self.sum_tree.total_priority
        if total <= 0.0:
            raise ValueError("No ")
        i = 0
        while i < batch_size:
            index, priority = self.sum_tree.sample()
            if priority < self.min_priority:
                continue  # skip uninitialized
            indices[i] = index
            weights[i] = (self.size * priority / total) ** -self.beta
            i += 1

        batch = self.buffer.get_batch(indices)
        batch.weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_annealing)

        return batch

    def update_priorities(self, indices: IntArray, td_errors: FloatArray) -> None:
        td_errors = np.clip(td_errors, a_min=self.min_priority, a_max=None)
        for index, td_error in zip(indices, td_errors):
            self.sum_tree.update_priority(index, td_error)
