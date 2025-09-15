import numpy as np
import pytest

from dqn.buffers import CircularBuffer
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
    """Create a CircularBuffer with small max size."""
    return CircularBuffer(max_size=5)


def _compare_experiences(exp1: Experience, exp2: Experience):
    assert np.allclose(exp1.state, exp2.state)
    assert np.allclose(exp1.next_state, exp2.next_state)
    assert exp1.action == exp2.action
    assert np.isclose(exp1.reward, exp2.reward)
    assert exp1.done == exp2.done


def test_add_single_experience(buffer, experience):
    buffer.add(experience)
    assert buffer.size == 1
    _compare_experiences(buffer.get(0), experience)


def test_add_multiple_experiences(buffer, experience):
    experiences = [experience] * 3
    for exp in experiences:
        buffer.add(exp)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        _compare_experiences(buffer.get(i), exp)


def test_add_batch_of_experiences(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        buf_exp = buffer.get(i)
        _compare_experiences(buf_exp, exp)


def test_buffer_max_size_overwrite(buffer, experience):
    experiences = [experience] * 10  # more than max_size
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == buffer._max_size
    # Oldest experiences are overwritten, last 5 remain
    for i, exp in enumerate(experiences[-buffer._max_size:]):
        _compare_experiences(buffer.get(i), exp)


def test_get_with_negative_index(buffer, experience):
    buffer.add(experience)
    assert buffer.get(-1) == experience
    with pytest.raises(IndexError):
        buffer.get(-2)  # out of bounds


def test_get_batch(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    indices = [0, 2]
    sub_batch = buffer.get_batch(indices)
    assert isinstance(sub_batch, ExperiencesBatch)
    assert sub_batch.size == len(indices)
    for i, idx in enumerate(indices):
        _compare_experiences(sub_batch.to_experiences()[i], buffer.get(idx))


def test_sample_returns_correct_batch(buffer, experience):
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
    # weights should be None
    assert getattr(batch, "weights", None) is None


def test_len_property(buffer, experience):
    assert buffer.size == 0
    buffer.add(experience)
    assert buffer.size == 1


def test_index_wrap_around(buffer, experience):
    # Fill buffer to capacity and add extra to wrap around
    for _ in range(buffer._max_size + 2):
        buffer.add(experience)
    assert buffer.size == buffer._max_size
    # Latest added experiences should be at correct positions
    for i in range(buffer._max_size):
        _compare_experiences(buffer.get(i), experience)
