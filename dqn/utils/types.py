from ast import In
from typing import (
    Protocol,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Union,
    runtime_checkable,
    Optional,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray, ArrayLike
import tensorflow as tf


@runtime_checkable
class SupportsBool(Protocol):
    """An ABC with one abstract method __bool__."""

    def __bool__(self) -> bool: ...


FloatLike = Union[float, np.floating, SupportsFloat]
IntLike = Union[int, np.integer, SupportsInt]
BoolLike = Union[bool, np.bool_, SupportsBool]

FloatArray = Union[NDArray[np.floating], Sequence[FloatLike]]
IntArray = Union[NDArray[np.integer], Sequence[IntLike]]
BoolArray = Union[NDArray[np.bool_], Sequence[BoolLike]]

SingleInput = Union[NDArray, tf.Tensor]
ModelInput = Union[SingleInput, Sequence[SingleInput]]


T = TypeVar("T", bound=np.generic)


def to_optional_array(x: Optional[ArrayLike], dtype: type[T]) -> NDArray[T] | None:
    if x is None:
        return None
    return np.asarray(x, dtype=dtype)
