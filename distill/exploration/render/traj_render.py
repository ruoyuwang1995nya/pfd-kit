from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

import dpdata
import numpy as np


if TYPE_CHECKING:
    from distill.exploration.selector import (
        ConfFilters,
    )


class TrajRender(ABC):
    @abstractmethod
    def get_confs(
        self,
        traj: List[Path],
        #id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        r"""Get configurations from trajectory by selection.

        Parameters
        ----------
        traj : List[Path]
            Trajectory files
        id_selected : List[List[int]]
            The selected frames. id_selected[ii][jj] is the jj-th selected frame
            from the ii-th trajectory. id_selected[ii] may be an empty list.
        type_map : List[str]
            The type map.
        conf_filters : ConfFilters
            Configuration filters
        optional_outputs : List[Path]
            Optional outputs of the exploration

        Returns
        -------
        ms:     dpdata.MultiSystems
            The configurations in dpdata.MultiSystems format
        """
        pass