from typing import Protocol

from ..experiences import Experience, ExperiencesBatch
from ..types import IntArray


class BufferProtocol(Protocol):
    """Protocol for a generic replay buffer."""

    @property
    def size(self) -> int: ...

    def add(self, exp: Experience) -> None: ...

    def add_batch(self, batch: ExperiencesBatch) -> None: ...

    def get(self, index: int) -> Experience: ...

    def get_batch(self, indices: IntArray) -> ExperiencesBatch: ...

    def sample(self, batch_size: int) -> ExperiencesBatch: ...
