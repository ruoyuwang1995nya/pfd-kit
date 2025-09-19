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

from ase import Atoms
from ase.io import read

from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from pfd.exploration.selector import (
        ConfFilters,
    )


@TrajRender.register("ase")
@TrajRender.register("calypso")
@TrajRender.register("calypso:merge")
class TrajRenderASE(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
    ):
        self.nopbc = nopbc

    def get_confs(
        self,
        trajs: List[Path],
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> List[Atoms]:
        ntraj = len(trajs)
        if optional_outputs:
            assert ntraj == len(optional_outputs)
        atoms_list=[]
        for ii in range(ntraj):
            ss = read(trajs[ii], index=':')
            if isinstance(ss,Atoms):
                ss=[ss]
            if self.nopbc:
                for atoms in ss:
                    atoms.pbc = not self.nopbc
            atoms_list.extend(ss)
        print(len(atoms_list))
        if conf_filters is not None:
            atoms_list = conf_filters.check(atoms_list)
        return atoms_list
