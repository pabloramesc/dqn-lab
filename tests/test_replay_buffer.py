import numpy as np
import pytest

from dqn.buffers import ReplayBuffer
from dqn.experiences import Experience, ExperiencesBatch


@pytest.fixture
def sample_experience():
    """Create a sample experience for testing."""
    return Experience(
        state=np.array([1.0, 2.0]),
        action=0,
        next_state=np.array([1.1, 2.1]),
        reward=1.0,
        done=False
    )


@pytest.fixture
def replay_buffer():
    """Create a ReplayBuffer with small max size."""
    return ReplayBuffer(max_size=5)


def test_add_single_experience(replay_buffer, sample_experience):
    replay_buffer.add(sample_experience)
    assert replay_buffer.size == 1
    assert replay_buffer.buffer[0] == sample_experience


def test_add_batch_of_experiences(replay_buffer, sample_experience):
    batch = [sample_experience] * 3
    replay_buffer.add_batch(batch)
    assert replay_buffer.size == 3
    for exp in batch:
        assert exp in replay_buffer.buffer


def test_buffer_max_size(replay_buffer, sample_experience):
    batch = [sample_experience] * 10  # more than max_size
    replay_buffer.add_batch(batch)
    assert replay_buffer.size == replay_buffer.max_size


def test_sample_returns_correct_batch(replay_buffer, sample_experience):
    # Fill buffer with 5 experiences
    for _ in range(5):
        replay_buffer.add(sample_experience)

    batch_size = 3
    batch: ExperiencesBatch = replay_buffer.sample(batch_size)
    
    assert isinstance(batch, ExperiencesBatch)
    assert batch.states.shape[0] == batch_size
    assert batch.actions.shape[0] == batch_size
    assert batch.next_states.shape[0] == batch_size
    assert batch.rewards.shape[0] == batch_size
    assert batch.dones.shape[0] == batch_size
    assert batch.indices.shape[0] == batch_size # type: ignore
    np.testing.assert_array_equal(batch.weights, None)


def test_sample_without_enough_experiences_raises(replay_buffer, sample_experience):
    replay_buffer.add(sample_experience)
    with pytest.raises(ValueError):
        # Trying to sample more than available
        replay_buffer.sample(2)


def test_len_property(replay_buffer, sample_experience):
    assert len(replay_buffer) == 0
    replay_buffer.add(sample_experience)
    assert len(replay_buffer) == 1
