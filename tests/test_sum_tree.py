import pytest
import numpy as np
from dqn.buffers.sum_tree import SumTree


@pytest.fixture
def tree():
    return SumTree(capacity=4)


def test_add_and_total_priority(tree):
    tree.add(1.0)
    assert tree.total_priority == 1.0

    tree.add(2.0)
    assert tree.total_priority == 3.0

    tree.add(3.0)
    tree.add(4.0)
    assert tree.total_priority == 10.0


def test_sample_single(tree):
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        tree.add(p)

    # Sample many times and check if results are within range
    for _ in range(100):
        data_idx, priority = tree.sample()
        assert 0 <= data_idx < tree.capacity
        assert priority > 0


def test_update_priority(tree):
    tree.add(1.0)
    tree.add(2.0)
    tree.add(3.0)
    tree.add(4.0)

    # Update second element
    tree.update_priority(1, 10.0)
    assert tree.total_priority == 1 + 10 + 3 + 4  # 18.0

    # Update first element
    tree.update_priority(0, 5.0)
    assert tree.total_priority == 5 + 10 + 3 + 4  # 22.0


def test_circular_add(tree):
    # Fill tree
    tree.add(1)
    tree.add(2)
    tree.add(3)
    tree.add(4)

    # Adding more should overwrite oldest
    tree.add(5)
    # total_priority = 5 + 2 + 3 + 4 = 14
    assert tree.total_priority == 14.0

    tree.add(6)
    # total_priority = 5 + 6 + 3 + 4 = 18
    assert tree.total_priority == 18.0


def test_sample_distribution(tree):
    # Add known priorities
    tree.add(1.0)
    tree.add(2.0)
    tree.add(3.0)
    tree.add(4.0)

    n_samples = 10_000
    counts = np.zeros(tree.capacity, dtype=int)
    for _ in range(n_samples):
        idx, _ = tree.sample()
        counts[idx] += 1

    # Probabilities proportional to priority
    expected_probs = np.array([1, 2, 3, 4]) / 10.0
    empirical_probs = counts / n_samples
    # Allow small error
    np.testing.assert_allclose(empirical_probs, expected_probs, rtol=0.1, atol=0.01)


def test_get_priority_after_add(tree):
    priorities = [0.1, 0.5, 0.9, 2.0]
    for p in priorities:
        tree.add(p)
    for i, p in enumerate(priorities):
        assert np.isclose(tree.get_priority(i), p)


def test_get_priority_after_update(tree):
    tree.add(0.1)
    tree.add(0.2)
    tree.add(0.3)

    # Update leaf 1
    tree.update_priority(1, 0.8)
    assert np.isclose(tree.get_priority(0), 0.1)
    assert np.isclose(tree.get_priority(1), 0.8)
    assert np.isclose(tree.get_priority(2), 0.3)
    assert np.isclose(tree.get_priority(3), 0.0)  # 4th leaf not set


def test_get_priority_with_circular_overwrite(tree):
    tree.add(0.1)
    tree.add(0.2)
    tree.add(0.3)
    tree.add(0.5)
    # Next add overwrites first leaf
    tree.add(0.9)
    assert np.isclose(tree.get_priority(0), 0.9)
    assert np.isclose(tree.get_priority(1), 0.2)
    assert np.isclose(tree.get_priority(2), 0.3)
    assert np.isclose(tree.get_priority(3), 0.5)


def test_check_consistency_true(tree):
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        tree.add(p)
    assert tree.check_consistency()


def test_check_consistency_false(tree):
    priorities = [1.0, 2.0, 3.0, 4.0]
    for p in priorities:
        tree.add(p)
    tree.tree[0] = 999.0  # Corrupt an internal node
    assert not tree.check_consistency()
