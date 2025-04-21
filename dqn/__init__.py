"""
The `dqn` module provides tools for implementing and training Deep Q-Networks (DQN).
It includes agents, exploration policies, replay buffers, and experience handling utilities.

Usage:
- Import `DQNAgent` to create and train a DQN agent.
- Use `EpsilonGreedyPolicy` or `BoltzmannPolicy` for exploration strategies.
- Store experiences in `ReplayBuffer` or `PriorityReplayBuffer`.
- Represent individual experiences with `Experience` and batches with `ExperiencesBatch`.
"""

from .dqn_agent import DQNAgent
from .exploration_policies import EpsilonGreedyPolicy, BoltzmannPolicy
from .replay_buffers import ReplayBuffer, PriorityReplayBuffer
from .experiences import Experience, ExperiencesBatch