from __future__ import (
    annotations,
)

from abc import (
    ABC,
    abstractmethod,
)

from ase import Atoms
import numpy as np
from typing import (
    List,
)

class ConfFilter(ABC):
    @abstractmethod
    def check(
        self,
        frame: Atoms,
    ) -> bool:
        """Check if the configuration is valid.

        Parameters
        ----------
        frame : dpdata.System
            A dpdata.System containing a single frame

        Returns
        -------
        valid : bool
            `True` if the configuration is a valid configuration, else `False`.

        """
        pass


class ConfFilters:
    """A list of ConfFilters"""

    def __init__(
        self,
    ):
        self._filters = []

    def add(
        self,
        conf_filter: ConfFilter,
    ) -> ConfFilters:
        self._filters.append(conf_filter)
        return self

    def check(
        self,
        conf: List[Atoms],
    ) -> List[Atoms]:
        selected_idx = np.arange(len(conf))
        for ff in self._filters:
            fsel = np.where([ff.check(conf[ii]) for ii in range(len(conf))])[0]
            selected_idx = np.intersect1d(selected_idx, fsel)
        return [conf[ii] for ii in selected_idx]
