"""
The `dqn` module provides tools for implementing and training Deep Q-Networks (DQN).
It includes agents, exploration policies, replay buffers, and experience handling utilities.

Usage:
- Import `DQNAgent` or `DQNAgentPER` to create and train a DQN agent.
- Use `EpsilonGreedyPolicy` or `BoltzmannPolicy` for exploration strategies.
- Represent individual experiences with `Experience` and batches with `ExperiencesBatch`.
"""

from .dqn_agent import DQNAgent
from .experiences import Experience, ExperiencesBatch
from .policies import BoltzmannPolicy, DecayType, EpsilonGreedyPolicy
from .rainbow_dqn import RainbowDQN
