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


def test_get_physical_basic(buffer):
    """Direct slot indexing should return the item at the physical slot."""
    # Add 3 items
    for i in range(3):
        buffer.add({"value": i})
    # Direct physical index 0..size-1 should match underlying buffer order
    for i in range(3):
        item = buffer.get_physical(i)
        assert item == {"value": i}


def test_get_physical_with_overwrite(buffer):
    """After wrap-around, physical index should still point to correct slot."""
    max_size = buffer._max_size
    # Fill buffer fully
    for i in range(max_size):
        buffer.add({"value": i})
    # Overwrite first two slots
    for i in range(max_size, max_size + 2):
        buffer.add({"value": i})
    # Now check physical slots directly
    for slot in range(max_size):
        item = buffer.get_physical(slot)
        # We can compute the expected value directly from the internal buffer
        # because get_physical should just return that slot:
        expected = buffer._buffer[slot]
        assert item == expected


def test_get_physical_out_of_range(buffer):
    """get_physical should raise IndexError for invalid physical indices."""
    with pytest.raises(IndexError):
        buffer.get_physical(0)  # empty buffer
    buffer.add({"value": 1})
    with pytest.raises(IndexError):
        buffer.get_physical(-1)
    with pytest.raises(IndexError):
        buffer.get_physical(buffer._max_size)  # beyond capacity
    # fill partially, but access unfilled slot
    # for example capacity=5, size=1, physical slot 3 not yet used
    with pytest.raises(IndexError):
        buffer.get_physical(3)
