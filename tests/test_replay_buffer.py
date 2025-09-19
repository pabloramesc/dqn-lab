import numpy as np
import pytest

from dqn.buffers import ReplayBuffer
from dqn.experiences import Experience, ExperiencesBatch


@pytest.fixture
def experience():
    """Create a sample experience for testing."""
    return Experience(
        state=np.array([1.0, 2.0]),
        action=0,
        next_state=np.array([1.1, 2.1]),
        reward=1.0,
        done=False,
    )


@pytest.fixture
def buffer():
    """Create a ReplayBuffer with small max size."""
    return ReplayBuffer(max_size=5)


def test_add_single_experience(buffer, experience):
    buffer.add(experience)
    assert buffer.size == 1
    assert buffer.get(0) == experience


def test_add_multiple_experiences(buffer, experience):
    experiences = [experience] * 3
    for exp in experiences:
        buffer.add(exp)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        assert exp is buffer.get(i)


def test_add_batch_of_experiences(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        buf_exp = buffer.get(i)
        # Check arrays
        assert np.allclose(exp.state, buf_exp.state)
        assert np.allclose(exp.next_state, buf_exp.next_state)
        # Check scalars
        assert exp.action == buf_exp.action
        assert np.isclose(exp.reward, buf_exp.reward)
        assert exp.done == buf_exp.done


def test_buffer_max_size(buffer, experience):
    experiences = [experience] * 10  # more than max_size
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == buffer.max_size


def test_sample_returns_correct_batch(buffer, experience):
    # Fill buffer with 5 experiences
    for _ in range(5):
        buffer.add(experience)

    batch_size = 3
    batch: ExperiencesBatch = buffer.sample(batch_size)

    assert isinstance(batch, ExperiencesBatch)
    assert batch.states.shape[0] == batch_size
    assert batch.actions.shape[0] == batch_size
    assert batch.next_states.shape[0] == batch_size
    assert batch.rewards.shape[0] == batch_size
    assert batch.dones.shape[0] == batch_size
    np.testing.assert_array_equal(batch.weights, None)
