from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

from . import (
    ConfFilters,
)


class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files."""

    @abstractmethod
    def select(
        self,
        # model_devis: List[Path],
        type_map: Optional[List[str]] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> List[Path]:
        pass
