import numpy as np
import pytest

from dqn.buffers import SimplePER
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
    """Create a PERBuffer with small max size."""
    return SimplePER(max_size=5, min_priority=0.1)


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
    assert buffer.priorities[0] == 0.9 + buffer.min_priority


def test_add_single_experience_without_td_error(buffer, experience):
    buffer.add(experience)
    assert buffer.size == 1
    assert buffer.get(0) == experience
    assert buffer.priorities[0] == 1.0 + buffer.min_priority  # default TD error value


def test_add_respects_min_priority(buffer, experience):
    buffer.add(experience, td_error=0.0)
    assert buffer.priorities[0] == buffer.min_priority


def test_add_negative_priority(buffer, experience):
    buffer.add(experience, td_error=-1.0)
    assert buffer.priorities[0] == 1.0 + buffer.min_priority


def test_add_multiple_experiences(buffer, experience):
    experiences = [experience] * 3
    td_errors = np.array([0.0, 0.1, 5.0])
    for i, exp in enumerate(experiences):
        buffer.add(exp, td_error=td_errors[i])
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        assert exp is buffer.get(i)
        expected_priority = abs(td_errors[i]) + buffer.min_priority
        assert np.isclose(buffer.priorities[i], expected_priority)


def test_add_batch_without_td_errors(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == 3
    for i, exp in enumerate(experiences):
        buf_exp = buffer.get(i)
        _compare_experiences(exp, buf_exp)
        # Check default TD error values
        assert buffer.priorities[i] == 1.0 + buffer.min_priority


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
        expected_priority = abs(td_errors[i]) + buffer.min_priority
        assert np.isclose(buffer.priorities[i], expected_priority)


def test_buffer_max_size(buffer, experience):
    experiences = [experience] * 10  # more than max_size
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    assert buffer.size == buffer.max_size


def test_sample_larger_than_buffer(buffer, experience):
    experiences = [experience] * 3
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch)
    sampled = buffer.sample(batch_size=buffer.size + 5)
    assert sampled.size == buffer.size + 5


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

    new_td_errors = np.array([0.9, 0.1, 0.0])
    buffer.update_priorities(np.array([0, 1, 2]), new_td_errors)

    expected = np.abs(new_td_errors) + buffer.min_priority
    for i in range(3):
        assert np.isclose(buffer.priorities[i], expected[i])


@pytest.mark.parametrize("alpha", np.linspace(0.0, 1.0, 5))
def test_sampling_probability_distribution(alpha, experience):
    buffer = SimplePER(max_size=5, alpha=alpha, min_priority=1e-6)
    experiences = [experience] * 5
    batch = ExperiencesBatch.from_experiences(experiences)
    td_errors = np.array([0.1, 0.5, 1.0, 2.0, 10.0])  # big difference in priorities
    buffer.add_batch(batch, td_errors=td_errors)

    n_samples = 10_000
    counts = np.zeros(buffer.size, dtype=int)
    for _ in range(n_samples):
        sampled = buffer.sample(batch_size=1)
        assert sampled.indices is not None
        idx = int(sampled.indices[0])
        counts[idx] += 1

    empirical_probs = counts / n_samples

    # Normalize expected distribution
    priorities = (np.abs(td_errors) + buffer.min_priority) ** buffer.alpha
    expected_probs = priorities / priorities.sum()

    # Check empirical distribution close to expected (within tolerance)
    assert np.allclose(empirical_probs, expected_probs, rtol=0.1, atol=0.01)


@pytest.mark.parametrize("beta", np.linspace(0.0, 1.0, 5))
def test_importance_sampling_weights(beta, experience):
    buffer = SimplePER(max_size=5, beta=beta, min_priority=1e-6)
    experiences = [experience] * 5
    td_errors = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch, td_errors=td_errors)

    sampled = buffer.sample(batch_size=5)
    weights = sampled.weights
    indices = sampled.indices
    assert weights is not None
    assert indices is not None

    # Calculate IS weights
    priorities = np.abs(td_errors) + buffer.min_priority
    probs = priorities**buffer.alpha / np.sum(priorities)
    expected_weights = (buffer.size * probs[indices]) ** -beta
    expected_weights /= expected_weights.max()

    assert np.allclose(weights, expected_weights, rtol=1e-6)


def test_beta_annealing(experience):
    buffer = SimplePER(max_size=5, beta=0.4, beta_annealing=1e-3)
    experiences = [experience] * buffer.max_size
    td_errors = np.random.normal(size=buffer.max_size)
    batch = ExperiencesBatch.from_experiences(experiences)
    buffer.add_batch(batch, td_errors=td_errors)

    # Test beta annealing for 100 steps
    for _ in range(100):
        _ = buffer.sample(batch_size=2)
    assert np.isclose(buffer.beta, 0.4 + 100 * 1e-3)
    
    # Test beta annealing to max
    for _ in range(1000):
        _ = buffer.sample(batch_size=2)
    assert np.isclose(buffer.beta, 1.0)
