import numpy as np
import pytest

from dqn.buffers import CircularBuffer


@pytest.fixture
def buffer() -> CircularBuffer[dict[str, int]]:
    """Create a CircularBuffer with small max size."""
    return CircularBuffer(max_size=5)


def test_add_single_item(buffer):
    item = {"Value": 42}
    buffer.add(item)
    assert buffer.size == 1
    assert buffer.get(0) == item


def test_add_multiple_items(buffer):
    items = [{"value": i} for i in range(3)]
    for item in items:
        buffer.add(item)
    assert buffer.size == 3
    for i, item in enumerate(items):
        assert buffer.get(i) == item


def test_buffer_overwrite_when_full(buffer):
    for i in range(10):
        buffer.add({"value": i})
    assert buffer.size == buffer._max_size
    # Last _max_size items should remain
    for i in range(buffer._max_size):
        expected = {"value": i + 10 - buffer._max_size}
        assert buffer.get(i) == expected


def test_get_with_negative_index(buffer):
    item = {"value": 42}
    buffer.add(item)
    assert buffer.get(-1) == item
    with pytest.raises(IndexError):
        buffer.get(-2)


def test_sample_returns_correct_size(buffer):
    for i in range(5):
        buffer.add({"value": i})
    batch_size = 3
    batch = buffer.sample(batch_size)
    assert len(batch) == batch_size
    for item in batch:
        assert isinstance(item, dict)


def test_clear_resets_buffer(buffer):
    item = {"value": 42}
    buffer.add(item)
    assert buffer.size == 1
    buffer.clear()
    assert buffer.size == 0
    with pytest.raises(IndexError):
        buffer.get(0)


def test_is_full_property(buffer):
    assert not buffer.is_full
    for i in range(buffer._max_size):
        buffer.add({"value": i})
    assert buffer.is_full


def test_index_wrap_around(buffer):
    for i in range(buffer._max_size + 2):
        buffer.add({"value": i})
    assert buffer.size == buffer._max_size
    for i in range(buffer._max_size):
        expected = {"value": i + 2}  # last items added
        assert buffer.get(i) == expected


def test_to_list_empty(buffer):
    l = buffer.to_list()
    assert isinstance(l, list)
    assert len(l) == 0


def test_to_list_not_full(buffer):
    for i in range(3):
        buffer.add({"value": i})
    l = buffer.to_list()
    for i in range(3):
        assert l[i] == buffer.get(i)


def test_to_list_full_wraparound(buffer):
    for i in range(buffer._max_size + 2):
        buffer.add({"value": i})
    l = buffer.to_list()
    for i in range(buffer._max_size):
        expected = {"value": i + 2}  # last items added
        assert l[i] == expected
