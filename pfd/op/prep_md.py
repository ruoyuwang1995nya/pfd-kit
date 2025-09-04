import json
import pickle
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)


from pfd.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)
from pfd.constants import ase_task_pattern

class PrepASE(OP):
    r"""Prepare the working directories for ASE tasks.

    A list of working directories (defined by `ip["task"]`)
    containing all files needed to start ASE tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "expl_task_grp": BigParameter(BaseExplorationTaskGroup),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `lmp_task_grp` : (`BigParameter(Path)`) Can be pickle loaded as a ExplorationTaskGroup. Definitions for LAMMPS tasks

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the LAMMPS simulation. The order fo the Paths should be consistent with `op["task_names"]`
        """

        ase_task_grp = ip["expl_task_grp"]
        cc = 0
        task_paths = []
        for tt in ase_task_grp:
            # ff: input files for each task
            ff = tt.files()
            tname = _mk_task_from_files(cc, ff)
            task_paths.append(tname)
            cc += 1
        task_names = [str(ii) for ii in task_paths]
        return OPIO(
            {
                "task_names": task_names,
                "task_paths": task_paths,
            }
        )

def _mk_task_from_files(cc, ff):
    """Write the input files to a task directory."""
    tname = Path(ase_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)
    for nn in ff.keys():
        (tname / nn).write_text(ff[nn])
    return tname