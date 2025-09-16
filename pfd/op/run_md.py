import glob
import json
import logging
import os
import random
import re
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

import ase
import numpy as np
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    TransientError,
)

from pfd.constants import (
    ase_conf_name,
    ase_input_name,
    ase_log_name,
    ase_traj_name
)

from pfd.exploration import md
from pfd.exploration.md import (
    MDRunner,
    CalculatorWrapper,
)

from pfd.utils import (
    BinaryFileInput,
    set_directory,
)



class RunASE(OP):
    r"""Execute a ASE MD task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The LAMMPS
    command is exectuted from directory `task_name`. The trajectory
    and the model deviation will be stored in files `op["traj"]` and
    `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
                "optional_output": Artifact(Path, optional=True),
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

            - `config`: (`dict`) The config of lmp task. Check `RunLmp.lmp_args` for definitions.
            - `task_name`: (`str`) The name of the task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepLmp`.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive molecular dynamics simulation.

        Returns
        -------
        Any
            Output dict with components:
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.

        Raises
        ------
        TransientError
            On the failure of LAMMPS execution. Handle different failure cases? e.g. loss atoms.
        """
        config = ip["config"] if ip["config"] is not None else {}
        ## what the config should be like?
        # {"calculator": "mace"}
        config = RunASE.normalize_config(config)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]
        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        model_files = [Path(ii).resolve() for ii in models]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)
            # instantiate calculator
            calc_style = config.pop("calculator", "mace")
            calc = CalculatorWrapper.get_calculator(calc_style)
            calc = calc().create(model_path=str(model_files[0]), **config)

            # instantiate MDRunner
            md_runner = MDRunner.from_file(
                filename=ase_conf_name
            )
            md_runner.set_calculator(calc)
            try:
                md_runner.run_md_from_json(
                    json_file=ase_input_name,
                )
            except Exception as e:
                raise TransientError(f"ASE MD/relax failed: {e}")
        ret_dict = {
            "log": work_dir / ase_log_name,
            "traj": work_dir / ase_traj_name
        }
        return OPIO(ret_dict)

    @staticmethod
    def ase_args():
        doc_calc_type = "The type of calculator to use, e.g., 'mace', 'mattersim'."
        return [
            Argument("calculator", str,  default="mattersim", doc=doc_calc_type, alias=['calc']),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunASE.ase_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data