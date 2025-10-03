import numpy as np
import pytest

from dqn.experiences import Experience, ExperiencesBatch


def test_experience_create_valid():
    state = np.array([1.0, 2.0])
    next_state = np.array([1.5, 2.5])
    exp = Experience.create(state, 1, next_state, 0.5, False, truncated=True)

    assert isinstance(exp, Experience)
    assert np.array_equal(exp.state, state)
    assert np.array_equal(exp.next_state, next_state)
    assert exp.action == 1
    assert exp.reward == 0.5
    assert not exp.done
    assert exp.truncated


def test_experience_create_invalid_shape():
    state = np.array([1.0, 2.0])
    next_state = np.array([1.5, 2.5, 3.0])
    with pytest.raises(ValueError):
        Experience.create(state, 0, next_state, 1.0, False)


def test_experience_create_invalid_dtype():
    state = np.array([1.0, 2.0], dtype=np.float32)
    next_state = np.array([1, 2], dtype=np.int32)
    with pytest.raises(ValueError):
        Experience.create(state, 0, next_state, 1.0, False)


def test_experiences_batch_creation_and_size():
    states = np.array([[1.0, 2.0], [3.0, 4.0]])
    next_states = np.array([[1.5, 2.5], [3.5, 4.5]])
    actions = np.array([0, 1])
    rewards = np.array([1.0, 2.0])
    dones = np.array([False, True])
    truncated = np.array([False, True])
    batch = ExperiencesBatch(
        states, actions, next_states, rewards, dones, truncated=truncated
    )

    assert batch.size == 2
    assert np.array_equal(batch.states, states)
    assert np.array_equal(batch.next_states, next_states)
    assert np.array_equal(batch.rewards, rewards)
    assert np.array_equal(batch.dones, dones)
    assert np.array_equal(batch.actions, actions)
    assert batch.truncated is not None
    assert np.array_equal(batch.truncated, truncated)

def test_to_experiences_includes_truncated():
    states = np.array([[1.0], [2.0]])
    next_states = np.array([[1.5], [2.5]])
    actions = np.array([0, 1])
    rewards = np.array([0.1, 0.2])
    dones = np.array([False, True])
    truncated = np.array([True, False])
    batch = ExperiencesBatch(
        states, actions, next_states, rewards, dones, truncated=truncated
    )

    exps = batch.to_experiences()
    assert len(exps) == 2
    assert exps[0].truncated is True
    assert exps[1].truncated is False
    assert exps[0].done is False
    assert exps[1].done is True


def test_from_experiences_round_trip():
    states = np.array([[1.0], [2.0]])
    next_states = np.array([[1.5], [2.5]])
    actions = np.array([0, 1])
    rewards = np.array([0.1, 0.2])
    dones = np.array([False, True])
    truncated = np.array([False, True])
    batch1 = ExperiencesBatch(
        states, actions, next_states, rewards, dones, truncated=truncated
    )

    exps = batch1.to_experiences()
    batch2 = ExperiencesBatch.from_experiences(exps)

    assert np.array_equal(batch1.states, batch2.states)
    assert np.array_equal(batch1.next_states, batch2.next_states)
    assert np.array_equal(batch1.actions, batch2.actions)
    assert np.array_equal(batch1.rewards, batch2.rewards)
    assert np.array_equal(batch1.dones, batch2.dones)
