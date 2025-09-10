import pytest
import numpy as np
from dqn.buffers.sum_tree import SumTree 


@pytest.fixture
def small_tree():
    return SumTree(capacity=4)


def test_add_and_total_priority(small_tree):
    small_tree.add(1.0)
    assert small_tree.total_priority == 1.0

    small_tree.add(2.0)
    assert small_tree.total_priority == 3.0

    small_tree.add(3.0)
    small_tree.add(4.0)
    assert small_tree.total_priority == 10.0


def test_sample_single(small_tree):
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        small_tree.add(p)

    # Sample many times and check if results are within range
    for _ in range(100):
        data_idx, priority = small_tree.sample()
        assert 0 <= data_idx < small_tree.capacity
        assert priority > 0


def test_update_priority(small_tree):
    small_tree.add(1.0)
    small_tree.add(2.0)
    small_tree.add(3.0)
    small_tree.add(4.0)

    # Update second element
    small_tree.update_priority(1, 10.0)
    assert small_tree.total_priority == 1 + 10 + 3 + 4  # 18.0

    # Update first element
    small_tree.update_priority(0, 5.0)
    assert small_tree.total_priority == 5 + 10 + 3 + 4  # 22.0


def test_circular_add(small_tree):
    # Fill tree
    small_tree.add(1)
    small_tree.add(2)
    small_tree.add(3)
    small_tree.add(4)

    # Adding more should overwrite oldest
    small_tree.add(5)
    # total_priority = 5 + 2 + 3 + 4 = 14
    assert small_tree.total_priority == 14.0

    small_tree.add(6)
    # total_priority = 5 + 6 + 3 + 4 = 18
    assert small_tree.total_priority == 18.0


def test_sample_distribution(small_tree):
    # Add known priorities
    small_tree.add(1.0)
    small_tree.add(2.0)
    small_tree.add(3.0)
    small_tree.add(4.0)

    counts = np.zeros(small_tree.capacity, dtype=int)
    n_samples = 10000

    for _ in range(n_samples):
        idx, _ = small_tree.sample()
        counts[idx] += 1

    # Probabilities proportional to priority
    expected_probs = np.array([1, 2, 3, 4]) / 10.0
    empirical_probs = counts / n_samples
    # Allow small error
    np.testing.assert_allclose(empirical_probs, expected_probs, atol=0.01)
