from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

FloatArray = Union[NDArray[np.floating], Sequence[float]]
IntArray = Union[NDArray[np.integer], Sequence[int]]
BoolArray = Union[NDArray[np.bool_], Sequence[bool]]
