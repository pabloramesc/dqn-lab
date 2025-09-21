import keras

from .buffers import NStepPER
from .dqn_agent import DQNAgent
from .experiences import ExperiencesBatch
from .policies import ExplorationPolicy

from typing import Optional

class RainbowDQN(DQNAgent):
    def __init__(
        self,
        model: keras.Model,
        policy: Optional[ExplorationPolicy] = None,
        batch_size: int = 32,
        memory_size: int = 100_000,
        update_freq: int = 10_000,
        gamma: float = 0.99,
        n_step: int = 3,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            policy=policy,
            batch_size=batch_size,
            memory_size=memory_size,
            gamma=gamma,
            update_freq=update_freq,
        )
        self.memory = NStepPER(
            max_size=memory_size,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            beta_annealing=beta_annealing,
        )

    def _train_interface(self, batch: ExperiencesBatch) -> dict:
        # q_values, td_errors = self._compute_targets(batch)
        q_values, td_errors = self._compute_targets_optimized(batch)
        metrics = self.model.train_on_batch(
            batch.states, q_values, sample_weight=batch.weights, return_dict=True
        )
        self.memory.update_priorities(batch.indices, td_errors)  # type: ignore
        return metrics
