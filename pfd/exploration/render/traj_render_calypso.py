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
    from pfd.exploration.selector import (
        ConfFilters,
    )


@TrajRender.register("calypso")
@TrajRender.register("calypso:default")
@TrajRender.register("calypso:merge")
class TrajRenderCalypso(TrajRender):
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
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        ntraj = len(trajs)
        if optional_outputs:
            assert ntraj == len(optional_outputs)

        traj_fmt = "ase/traj"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            traj_dir = trajs[ii]
            traj_files = list(traj_dir.rglob("*.traj"))
            for traj_file in traj_files:
                ss = dpdata.System(traj_file, fmt=traj_fmt, type_map=type_map)
                ss.nopbc = self.nopbc
                ms.append(ss)
        if conf_filters is not None:
            ms2 = dpdata.MultiSystems(type_map=type_map)
            for s in ms:
                s2 = conf_filters.check(s)
                if len(s2) > 0:
                    ms2.append(s2)
            ms = ms2
        return ms
