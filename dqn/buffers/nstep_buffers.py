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
        """Prioritized Experience Replay buffer with n-step returns.

        Args:
            max_size: Maximum size of the replay buffer.
            n_step: Number of steps to accumulate for n-step return.
            gamma: Discount factor for n-step return.
            alpha: PER exponent controlling how much prioritization is used.
            beta: Initial importance-sampling exponent.
            beta_annealing: Increment per step for beta.
            min_priority: Minimum priority to avoid zero sampling probability.
        """
        super().__init__(max_size, alpha, beta, beta_annealing, min_priority)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffers: dict[int, deque[Experience]] = {}

    def add(self, exp: Experience, td_error: float = 1.0, agent_id: int = 0) -> None:
        """Add an experience to the n-step buffer and store a new n-step transition if ready.

        - Maintains a separate buffer per agent.
        - Once the buffer length reaches n, adds the aggregated n-step experience to the PER buffer.
        - If the added experience marks the end of an episode (done=True),
          flushes the remaining partial n-step buffer.

        Args:
            exp: The single-step experience to add.
            td_error: Initial TD error/priority for the new transition.
            agent_id: ID of the agent (for multi-agent setups).
        """
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
            self.flush(agent_id, td_error)

    def add_batch(
        self,
        batch: ExperiencesBatch,
        td_errors: FloatArray | None = None,
        agent_ids: IntArray | None = None,
    ) -> None:
        """Add a batch of experiences from different agents to the buffer.
        
        - If `agent_ids` is provided, each entry is assigned to the corresponding
        agent's buffer.        
        - If `agent_ids` is `None`, the agent IDs are generated automatically as
        [0, 1, ..., N-1], where N is the number of experiences in the batch.

        Args:
            batch: Batch of experiences to add (converted internally to list of Experience).
            td_errors: Optional array of TD errors for each experience. Defaults to 1.0.
            agent_ids: Optional array of agent IDs corresponding to each experience.
        """
        experiences = batch.to_experiences()
        for i, exp in enumerate(experiences):
            agent_id = agent_ids[i] if agent_ids is not None else i
            td_error = td_errors[i] if td_errors is not None else 1.0
            self.add(exp, td_error, agent_id)

    def flush(self, agent_id: int | None = None, td_error: float = 1.0) -> None:
        """Flush one agent's n-step buffer or all buffers.

        Args:
            agent_id: ID of the agent to flush. If None, flush all agents.
            td_error: Priority to assign to remaining experiences.
        """
        if agent_id is None:
            # Flush all buffers
            for aid in list(self.n_step_buffers.keys()):
                self._flush_single(aid, td_error)
        else:
            # Flush only this agent's buffer
            self._flush_single(agent_id, td_error)

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

    def _flush_single(self, agent_id: int, td_error: float) -> None:
        """Flush a single agent's n-steps buffer."""
        buffer = self.n_step_buffers.get(agent_id)
        if not buffer:
            return  # nothing to flush
        while buffer:
            n_step_exp = self._get_n_step_experience(buffer)
            super().add(n_step_exp, td_error)
            buffer.popleft()
