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
)

from ase import Atoms

if TYPE_CHECKING:
    from pfd.exploration.selector import (
        ConfFilters,
    )


class TrajRender(ABC):
    __RenderTypes = {}

    @abstractmethod
    def get_confs(
        self,
        traj: List[Path],
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> List['Atoms']:
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

    @staticmethod
    def register(key: str):
        """Register a traj render. Used as decorators

        Args:
            key (str): key of the traj
        """

        def decorator(object):
            TrajRender.__RenderTypes[key] = object
            return object

        return decorator

    @staticmethod
    def get_driver(key: str):
        """Get a driver for ModelEval

        Args:
            key (str): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            return TrajRender.__RenderTypes[key]
        except KeyError as e:
            raise RuntimeError("unknown driver: " + key) from e

    @staticmethod
    def get_drivers() -> dict:
        """Get all drivers

        Returns:
            dict: all drivers
        """
        return TrajRender.__RenderTypes
