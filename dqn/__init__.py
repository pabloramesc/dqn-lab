"""
The `dqn` module provides tools for implementing and training Deep Q-Networks (DQN).
It includes agents, exploration policies, replay buffers, and experience handling utilities.

Usage:
- Import `DQNAgent` or `DQNAgentPER` to create and train a DQN agent.
- Use `EpsilonGreedyPolicy` or `BoltzmannPolicy` for exploration strategies.
- Represent individual experiences with `Experience` and batches with `ExperiencesBatch`.
"""

from .dqn_agent import DQNAgent
from .dqn_agent_per import DQNAgentPER
from .dqn_trainer import DQNTrainer
from .experiences import Experience, ExperiencesBatch
from .policies import BoltzmannPolicy, EpsilonGreedyPolicy
