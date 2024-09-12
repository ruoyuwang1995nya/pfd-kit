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

from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from distill.exploration.selector import (
        ConfFilters,
    )

class TrajRenderLammps(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
        use_ele_temp: int = 0,
    ):
        self.nopbc = nopbc
        self.use_ele_temp = use_ele_temp

    def get_confs(
        self,
        trajs: List[Path],
        #id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        ntraj = len(trajs)
        if optional_outputs:
            assert ntraj == len(optional_outputs)

        traj_fmt = "lammps/dump"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            #if len(id_selected[ii]) > 0:
            ss = dpdata.System(trajs[ii], fmt=traj_fmt, type_map=type_map)
            ss.nopbc = self.nopbc
            #ss = ss.sub_system(id_selected[ii])
            ms.append(ss)
        if conf_filters is not None:
            ms2 = dpdata.MultiSystems(type_map=type_map)
            for s in ms:
                s2 = conf_filters.check(s)
                if len(s2) > 0:
                    ms2.append(s2)
            ms = ms2
        return ms