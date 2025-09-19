import numpy as np
import pytest

from dqn.buffers import OptimizedPER
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
    """Create a OptimizedPER with small max size."""
    return OptimizedPER(max_size=5, min_priority=0.1)


def _compare_experiences(exp1: Experience, exp2: Experience):
    # Check arrays
    assert np.allclose(exp1.state, exp2.state)
    assert np.allclose(exp1.next_state, exp2.next_state)
    # Check scalars
    assert exp1.action == exp2.action
    assert np.isclose(exp1.reward, exp2.reward)
    assert exp1.done == exp2.done


def test_add_single_experience(buffer, experience):
    buffer.add(experience, td_error=0.9)
    assert buffer.size == 1
    assert buffer.get(0) == experience
    assert np.isclose(buffer.sum_tree.total_priority, 0.9)


def test_add_single_experience_without_td_error(buffer, experience):
    buffer.add(experience)
    assert buffer.size == 1
    assert buffer.get(0) == experience
    assert np.isclose(buffer.sum_tree.total_priority, 1.0)  # default TD error value


def test_add_respects_min_priority(buffer, experience):
    buffer.add(experience, td_error=0.0)
    priority = buffer.sum_tree.total_priority
    assert np.isclose(priority, buffer.min_priority)


def test_add_multiple_experiences(buffer, experience):
    experiences = [experience] * 3
    td_errors = np.array([0.0, 0.1, 5.0])
    for i, exp in enumerate(experiences):
        buffer.add(exp, td_error=td_errors[i])
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        assert exp is buffer.get(i)
        priority = buffer.sum_tree.get_priority(i)
        expected = max(td_errors[i], priority)
        assert np.isclose(priority, expected)


def test_add_batch_without_td_errors(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        buf_exp = buffer.get(i)
        _compare_experiences(exp, buf_exp)
        # Check default TD error values
        priority = buffer.sum_tree.get_priority(i)
        assert np.isclose(priority, 1.0)


def test_add_batch_with_td_errors(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    td_errors = np.array([0.0, 0.1, 5.0])
    buffer.add_batch(batch, td_errors=td_errors)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        buf_exp = buffer.get(i)
        _compare_experiences(exp, buf_exp)
        # Check default TD error values
        priority = buffer.sum_tree.get_priority(i)
        expected = max(td_errors[i], priority)
        assert np.isclose(priority, expected)


def test_buffer_max_size(buffer, experience):
    experiences = [experience] * 10  # more than max_size
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == buffer.max_size


def test_sample_returns_weighted_batch(buffer, experience):
    experiences = [experience] * 5
    batch = ExperiencesBatch.from_experiences(experiences)
    td_errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    buffer.add_batch(batch, td_errors)

    sampled = buffer.sample(batch_size=3)

    assert isinstance(sampled, ExperiencesBatch)
    assert sampled.size == 3
    assert sampled.weights is not None
    assert len(sampled.weights) == 3
    assert np.all(sampled.weights > 0)
    assert np.all(sampled.weights <= 1.0)

    # beta should increase due to annealing
    assert buffer.beta > 0.4


def test_update_priorities(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch, td_errors=np.array([0.5, 0.6, 0.7]))

    indices = np.array([0, 1, 2])
    new_td_errors = np.array([0.9, 0.1, 0.0])
    buffer.update_priorities(indices, new_td_errors)

    for i, td in zip(indices, new_td_errors):
        priority = buffer.sum_tree.get_priority(i)
        expected = max(td, buffer.min_priority)
        assert np.isclose(priority, expected)


def test_sampling_probability_distribution(buffer, experience):
    buffer.alpha = 1.0  # alpha=1 → pure proportional sampling
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    td_errors = np.array([0.1, 1.0, 10.0])  # big difference in priorities
    buffer.add_batch(batch, td_errors=td_errors)

    n_samples = 10_000
    counts = np.zeros(buffer.size, dtype=int)
    for _ in range(n_samples):
        sampled = buffer.sample(batch_size=1)
        idx = int(sampled.indices[0])
        counts[idx] += 1

    empirical_probs = counts / n_samples

    # Normalize expected distribution
    priorities = (
        np.clip(td_errors, buffer.min_priority, None) ** buffer.alpha
        + buffer.min_priority
    )
    expected_probs = priorities / priorities.sum()

    # Check empirical distribution close to expected (within tolerance)
    assert np.allclose(empirical_probs, expected_probs, rtol=0.1, atol=0.01)
