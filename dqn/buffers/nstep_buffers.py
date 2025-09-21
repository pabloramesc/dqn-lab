from collections import deque

from dqn.experiences import Experience, ExperiencesBatch

from ..experiences import Experience
from ..utils.types import FloatArray, IntArray
from .optimized_per import OptimizedPER


class NStepPER(OptimizedPER):
    def __init__(
        self,
        max_size: int,
        n_step: int,
        gamma: float,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 1e-6,
        min_priority: float = 1e-6,
    ) -> None:
        super().__init__(max_size, alpha, beta, beta_annealing, min_priority)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffers: dict[int, deque[Experience]] = {}

    def add(self, exp: Experience, td_error: float = 1.0, agent_id: int = 0) -> None:
        """Add an experience and create n-step transitions."""
        if agent_id not in self.n_step_buffers:
            self.n_step_buffers[agent_id] = deque(maxlen=self.n_step)

        buffer = self.n_step_buffers[agent_id]
        buffer.append(exp)

        # If buffer has enough steps, create n-step transition
        if len(buffer) == self.n_step:
            n_step_exp = self._get_n_step_experience(buffer)
            super().add(n_step_exp, td_error)

        # If this experience is terminal, flush n-step buffer
        if exp.done:
            pass

    def add_batch(
        self,
        batch: ExperiencesBatch,
        td_errors: FloatArray | None = None,
        agent_ids: IntArray | None = None,
    ) -> None:
        experiences = batch.to_experiences()
        for i, exp in enumerate(experiences):
            agent_id = agent_ids[i] if agent_ids is not None else i
            td_error = td_errors[i] if td_errors is not None else 1.0
            self.add(exp, td_error, agent_id)

    def _get_n_step_experience(self, buffer: deque[Experience]) -> Experience:
        """Compute n-step return and create a new experience."""
        first_exp = buffer[0]
        last_exp = buffer[-1]

        R = 0.0
        for i, exp in enumerate(buffer):
            R += (self.gamma**i) * exp.reward
            if exp.done:
                last_exp = exp
                break  # Stop accumulating if episode ends

        return Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=R,
            next_state=last_exp.next_state,
            done=last_exp.done,
        )
