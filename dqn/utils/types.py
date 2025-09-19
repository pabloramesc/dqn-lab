from typing import (
    Protocol,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SupportsBool(Protocol):
    """An ABC with one abstract method __bool__."""

    def __bool__(self) -> bool: ...


FloatLike = float | np.floating
IntLike = int | np.integer
BoolLike = bool | np.bool_

FloatArray = Union[NDArray[np.floating], Sequence[float]]
IntArray = Union[NDArray[np.integer], Sequence[int]]
BoolArray = Union[NDArray[np.bool_], Sequence[bool]]
