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
    Parameter,
)

from pfd.constants import (
    calypso_check_opt_file,
    calypso_input_file,
    calypso_run_opt_file,
    calypso_task_pattern,
    ase_input_name,
    model_name_pattern,
)
from pfd.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)
from pfd.utils import (
    set_directory,
)



# the first node of the workflow
class PrepCalyInput(OP):
    r"""Prepare the working directories and input file for generating structures.

    A calypso input file will be generated according to the given parameters
    (defined by `ip["caly_inputs"]`). The artifact will be return
    (ip[`input_files`]). The name of directory is `ip["task_names"]`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "caly_task_grp": BigParameter(
                    BaseExplorationTaskGroup
                ),  # calypso input params
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "ntasks": Parameter(int),
                "task_names": BigParameter(List[str]),  # task dir names
                "input_dat_files": Artifact(List[Path]),  # `input.dat`s
                "caly_run_opt_files": Artifact(List[Path]),
                "caly_check_opt_files": Artifact(List[Path]),
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
            - `caly_task_grp` : (`BigParameter()`) Definitions for CALYPSO input file.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of CALYPSO tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `input_dat_files`: (`Artifact(List[Path])`) The parepared working paths of the task containing input files (`input.dat` and `calypso_run_opt.py`) needed to generate structures by CALYPSO and make structure optimization with DP model.
            - `caly_run_opt_files`: (`Artifact(List[Path])`)
            - `caly_check_opt_files`: (`Artifact(List[Path])`)
        """

        cc = 0
        task_paths = []
        input_dat_files = []
        caly_run_opt_files = []
        caly_check_opt_files = []
        caly_task_grp = ip["caly_task_grp"]
        for tt in caly_task_grp:
            ff = tt.files()
            tname = _mk_task_from_files(cc, ff)
            task_paths.append(tname)
            input_dat_files.append(tname / calypso_input_file)
            caly_run_opt_files.append(tname / ase_input_name)
            caly_check_opt_files.append(tname / calypso_check_opt_file)
            cc += 1
        task_names = [str(ii) for ii in task_paths]

        return OPIO(
            {
                "ntasks": len(task_names),
                "task_names": task_names,
                "input_dat_files": input_dat_files,
                "caly_run_opt_files": caly_run_opt_files,
                "caly_check_opt_files": caly_check_opt_files,
            }
        )


def _mk_task_from_files(cc, ff):
    tname = Path(calypso_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)
    for file_name, file_content in ff.items():
        (tname / file_name).write_text(file_content)
    return tname