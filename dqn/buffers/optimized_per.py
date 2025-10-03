import numpy as np
from numpy.typing import NDArray

from dqn.experiences import Experience, ExperiencesBatch

from .simple_per import SimplePER
from .sum_tree import SumTree
from ..utils.types import IntArray, FloatArray


class OptimizedPER(SimplePER):

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
        self.priorities = None

    def add(self, exp: Experience, td_error: float = 1.0) -> None:
        self.buffer.add(exp)
        priority = abs(td_error) + self.min_priority
        self.sum_tree.add(priority**self.alpha)

    def add_batch(
        self, batch: ExperiencesBatch, td_errors: FloatArray | None = None
    ) -> None:
        experiences = batch.to_experiences()
        for exp in experiences:
            self.buffer.add(exp)

        if td_errors is None:
            td_errors = np.ones(batch.size)

        priorities = np.abs(td_errors) + self.min_priority

        for p in priorities:
            self.sum_tree.add(p**self.alpha)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        indices = np.zeros(batch_size, dtype=int)
        weights = np.zeros(batch_size, dtype=np.float32)

        total = self.sum_tree.total_priority
        if total < self.min_priority:
            raise RuntimeError("Total priority must be greater than min priority.")

        i = 0
        while i < batch_size:
            index, priority = self.sum_tree.sample()
            if index >= self.size:
                continue  # skip uninitialized
            indices[i] = index
            weights[i] = (self.size * priority / total) ** -self.beta
            i += 1

        experiences = [self.buffer.get_physical(i) for i in indices]
        batch = ExperiencesBatch.from_experiences(experiences, indices)
        batch.weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_annealing)

        return batch

    def update_priorities(self, indices: IntArray, td_errors: FloatArray) -> None:
        priorities = (np.abs(td_errors) + self.min_priority)**self.alpha
        for idx, p in zip(indices, priorities):
            self.sum_tree.update_priority(idx, p)
