import numpy as np
import pytest
from dqn.buffers import NumpyBuffer


@pytest.fixture
def scalar_buffer():
    """A small scalar buffer with max_size 3."""
    return NumpyBuffer(max_size=3, shape=(), dtype=np.float32)


@pytest.fixture
def vector_buffer():
    """A small 2-element vector buffer with max_size 2."""
    return NumpyBuffer(max_size=2, shape=(2,), dtype=np.float32)


def test_add_and_get_scalar(scalar_buffer):
    buf = scalar_buffer
    buf.add(1.0)
    buf.add(2.0)
    assert buf.size == 2
    assert not buf.is_full
    assert buf.get(0) == 1.0
    assert buf.get(1) == 2.0
    assert buf[-1] == 2.0

    buf.set(0, 10.0)
    assert buf.get(0) == 10.0


def test_wrap_around_behavior(scalar_buffer):
    buf = scalar_buffer
    buf.add(1.0)
    buf.add(2.0)
    buf.add(3.0)
    assert buf.is_full

    buf.add(4.0)
    assert buf.size == 3
    assert buf.get(0) == 2.0
    assert buf.get(1) == 3.0
    assert buf.get(2) == 4.0


def test_to_array_order(scalar_buffer):
    buf = scalar_buffer
    buf.add(1.0)
    buf.add(2.0)
    np.testing.assert_array_equal(buf.to_array(), np.array([1.0, 2.0]))

    buf.add(3.0)
    buf.add(4.0)
    np.testing.assert_array_equal(buf.to_array(), np.array([2.0, 3.0, 4.0]))


def test_multidimensional_buffer(vector_buffer):
    buf = vector_buffer
    buf.add(np.array([1.0, 2.0]))
    buf.add(np.array([3.0, 4.0]))

    np.testing.assert_array_equal(buf.get(0), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(buf.get(1), np.array([3.0, 4.0]))

    buf.add(np.array([5.0, 6.0]))
    np.testing.assert_array_equal(buf.get(0), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(buf.get(1), np.array([5.0, 6.0]))


def test_index_errors(scalar_buffer):
    buf = scalar_buffer
    with pytest.raises(IndexError):
        buf.get(0)

    buf.add(1.0)
    with pytest.raises(IndexError):
        buf.get(2)
    with pytest.raises(IndexError):
        buf.set(2, 0.0)


def test_shape_validation(vector_buffer):
    buf = vector_buffer
    with pytest.raises(ValueError):
        buf.add(np.array([1.0]))  # wrong shape
    buf.add(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        buf.set(0, np.array([1.0]))  # wrong shape
