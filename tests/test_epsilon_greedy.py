import numpy as np
import pytest

from dqn.policies import EpsilonGreedyPolicy


def test_initialization():
    policy = EpsilonGreedyPolicy(epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.01, decay_type="linear")
    assert policy.epsilon == 0.5
    assert policy.epsilon_min == 0.1
    assert policy.epsilon_decay == 0.01
    assert policy.decay_type == "linear"


def test_select_action_returns_valid_index():
    q_values = np.array([0.1, 0.5, 0.2])
    policy = EpsilonGreedyPolicy(epsilon=0.0)  # always greedy
    action = policy.select_action(q_values)
    assert action == np.argmax(q_values)


def test_select_action_with_full_exploration():
    q_values = np.array([0.1, 0.5, 0.2])
    policy = EpsilonGreedyPolicy(epsilon=1.0)  # always random
    actions = set(policy.select_action(q_values) for _ in range(50))
    assert actions.issubset({0, 1, 2})
    assert len(actions) > 1  # ensure randomness


def test_select_action_batch_shape_and_values():
    q_values = np.array([[0.1, 0.5], [0.7, 0.3], [0.2, 0.8]])
    policy = EpsilonGreedyPolicy(epsilon=0.0)  # always greedy
    actions = policy.select_action_batch(q_values)
    assert actions.shape[0] == q_values.shape[0]
    np.testing.assert_array_equal(actions, np.argmax(q_values, axis=1))


def test_update_params_linear_decay():
    policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.1, decay_type="linear")
    policy.update_params(steps=5)
    expected = max(0.0, 1.0 - 0.1 * 5)
    assert policy.epsilon == expected


def test_update_params_exponential_decay():
    policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.9, decay_type="exponential")
    policy.update_params(steps=2)
    expected = max(0.0, 1.0 * 0.9**2)
    assert np.isclose(policy.epsilon, expected)


def test_update_params_fixed_decay_does_nothing():
    policy = EpsilonGreedyPolicy(epsilon=0.5, decay_type="fixed")
    policy.update_params(steps=10)
    assert policy.epsilon == 0.5


def test_update_params_invalid_decay_type_raises():
    policy = EpsilonGreedyPolicy(epsilon=0.5, decay_type="invalid") # type: ignore
    with pytest.raises(ValueError):
        policy.update_params()


def test_get_dynamic_params():
    policy = EpsilonGreedyPolicy(epsilon=0.42)
    params = policy.get_dynamic_params()
    assert isinstance(params, dict)
    assert params["epsilon"] == 0.42


def test_set_full_exploration():
    policy = EpsilonGreedyPolicy(epsilon=0.2)
    policy.set_full_exploration()
    assert policy.epsilon == 1.0
    assert policy.decay_type == "fixed"


def test_set_full_exploitation():
    policy = EpsilonGreedyPolicy(epsilon=0.8)
    policy.set_full_exploitation()
    assert policy.epsilon == 0.0
    assert policy.decay_type == "fixed"
