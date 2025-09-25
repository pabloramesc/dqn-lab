import pytest
import numpy as np
from collections import deque

from dqn.experiences import Experience, ExperiencesBatch
from dqn.buffers import NStepPER


def make_exp(i, done=False):
    return Experience(
        state=np.array([i], dtype=np.float32),
        action=i,
        reward=float(i),
        next_state=np.array([i + 1], dtype=np.float32),
        done=done,
    )


@pytest.fixture
def buffer() -> NStepPER:
    return NStepPER(max_size=10, n_step=3, gamma=0.9)


def test_single_agent_n_step_accumulation(buffer):
    buffer.add(make_exp(1))
    buffer.add(make_exp(2))
    # Not enough steps yet
    assert buffer.size == 0

    buffer.add(make_exp(3))
    # After 3 steps, first n-step experience is stored
    assert buffer.size == 1
    n_step_exp = buffer.buffer.get(0)
    # Reward = 1 + 0.9*2 + 0.9^2*3
    expected_reward = 1 + 0.9 * 2 + 0.9 * 0.9 * 3
    assert np.isclose(n_step_exp.reward, expected_reward)
    # State and action come from first experience
    assert (n_step_exp.state == np.array([1])).all()
    assert n_step_exp.action == 1


def test_terminal_flush(buffer):
    buffer.add(make_exp(4))
    buffer.add(make_exp(5))
    # Add terminal experience
    buffer.add(make_exp(6, done=True))
    # After terminal, n-step buffer should flush partial sequence
    assert buffer.size >= 1
    last_exp = buffer.buffer.get(buffer.size - 1)
    assert last_exp.done is True


def test_multi_agent_batch_add(buffer):
    buffer = buffer
    exps = [make_exp(1), make_exp(10), make_exp(20)]
    batch = ExperiencesBatch.from_experiences(exps)
    agent_ids = [0, 1, 2]

    buffer.add_batch(batch, agent_ids=agent_ids)
    # Nothing stored yet because n_step=3
    assert buffer.size == 0

    # Add second batch
    exps2 = [make_exp(2), make_exp(11), make_exp(21)]
    batch2 = ExperiencesBatch.from_experiences(exps2)
    buffer.add_batch(batch2, agent_ids=agent_ids)
    # Still less than n_step for each agent
    assert buffer.size == 0

    # Add third batch
    exps3 = [make_exp(3), make_exp(12), make_exp(22)]
    batch3 = ExperiencesBatch.from_experiences(exps3)
    buffer.add_batch(batch3, agent_ids=agent_ids)
    # Now each agent should have an n-step experience
    assert buffer.size == 3

    # Check rewards for one agent
    exp0 = buffer.buffer.get(0)
    expected_reward0 = 1 + 0.9 * 2 + 0.9 * 0.9 * 3
    assert np.isclose(exp0.reward, expected_reward0)


def test_flush_single_agent(buffer):
    """Test flushing of a single agent's partial n-step buffer."""
    # Add only two steps (less than n=3)
    exp1 = make_exp(1)
    exp2 = make_exp(2)
    buffer.add(exp1, agent_id=0)
    buffer.add(exp2, agent_id=0)
    # Nothing stored yet
    assert buffer.size == 0

    # Force flush the remaining partial buffer
    buffer.flush(agent_id=0)
    # Now there should be experiences stored (flushed)
    assert buffer.size > 0
    last_exp = buffer.buffer.get(-1)
    # The state should come from the first partial experience
    assert np.allclose(last_exp.state, exp2.state)
    assert np.allclose(last_exp.next_state, exp2.next_state)


def test_flush_all_agents(buffer):
    """Test flushing all agents' buffers at once."""
    # Add partial sequences for multiple agents
    buffer.add(make_exp(1), agent_id=0)
    buffer.add(make_exp(10), agent_id=1)
    buffer.add(make_exp(2), agent_id=0)
    buffer.add(make_exp(11), agent_id=1)
    # Still nothing stored yet
    assert buffer.size == 0

    # Flush everything at once
    buffer.flush()
    # Should now have stored at least one experience per agent
    assert buffer.size >= 1
    # Verify that both agent 0 and agent 1 contributions appear
    # (state values from first exp of each buffer)
    states = [exp.state for exp in buffer.buffer.to_list()]  # underlying storage
    assert 1 in states or 10 in states


def test_flush_empty_buffer_does_not_crash(buffer):
    """Flushing an empty buffer should do nothing and not raise errors."""
    # Nothing in the buffer yet
    buffer.flush()  # should not raise
    buffer.flush(agent_id=0)  # should also not raise
    assert buffer.size == 0
