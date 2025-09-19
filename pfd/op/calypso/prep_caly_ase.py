import json
import logging
from math import log
import pickle
import shutil
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
    TransientError,
)

from pfd.constants import (
    calypso_check_opt_file,
    calypso_opt_dir_name,
    calypso_run_opt_file,
    model_name_pattern,
)
from pfd.exploration.task import (
    ExplorationTaskGroup,
)
from pfd.utils import (
    BinaryFileInput,
    set_directory,
)
from pfd.utils.run_command import (
    run_command,
)
from pfd.constants import ase_conf_name

import ase
from ase.io import read,write
import logging

# prepare working directories for calypso structure optimization with ase calculator
class PrepCalyASEOptim(OP):
    r"""Prepare the working directories and input file according to slices information
    for structure optimization with ASE.

    `POSCAR_*`, `frozen_model.pb` or `model.ckpt.pt`, `calypso_run_opt.py`
    and `calypso_check_opt.py` will be copied or symlink to each optimization directory
    from `ip["work_path"]`, according to the group size of ip["template_slice_config"].
    The POSCAR_* will be splited into group_size parts and the name of each path will be returned
    in a `task_names` list and `task_dirs` list.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_name": BigParameter(str),  # calypso_task.idx
                "finished": Parameter(str),
                "template_slice_config": Parameter(dict),
                "poscar_dir": Artifact(
                    Path
                ),  # from run_calypso first, then from collect_run_caly
                "models_dir": Artifact(Path),  #
                "caly_run_opt_file": Artifact(Path),  # from prep_caly_input
                "caly_check_opt_file": Artifact(Path),  # from prep_caly_input
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": Parameter(List[str]),
                "task_dirs": Artifact(List[Path]),
                "caly_run_opt_file": Artifact(Path),  # from prep_caly_input
                "caly_check_opt_file": Artifact(Path),  # from prep_caly_input
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
            - `task_name` : (`str`)
            - `finished` : (`str`)
            - `template_slice_config` : (`dict`)
            - `poscar_dir` : (`Path`)
            - `models_dir` : (`Path`)
            - `caly_run_opt_file` : (`Path`)
            - `caly_check_opt_file` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`)
            - `task_dirs`: (`Artifact(List[Path])`)
            - `caly_run_opt_file` : (`Path`)
            - `caly_check_opt_file` : (`Path`)

        """
        finished = ip["finished"]

        work_dir = Path(ip["task_name"])
        poscar_dir = ip["poscar_dir"]
        
        # models_dir = ip["models_dir"]
        _caly_run_opt_file = ip["caly_run_opt_file"]
        _caly_check_opt_file = ip["caly_check_opt_file"]
        caly_run_opt_file = _caly_run_opt_file.resolve()
        caly_check_opt_file = _caly_check_opt_file.resolve()
        poscar_list = [poscar.resolve() for poscar in poscar_dir.rglob("POSCAR_*")]
        poscar_list = sorted(poscar_list, key=lambda x: int(x.name.strip("POSCAR_")))
        group_size = ip["template_slice_config"].get("group_size", len(poscar_list))
        if group_size is None:
            group_size = len(poscar_list)
        
        logging.info(f"Group size: {group_size}, total POSCAR files: {len(poscar_list)}")

        with set_directory(work_dir):
            Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
            Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)
            if finished == "false":
                grouped_poscar_list = [
                    poscar_list[i : i + group_size]
                    for i in range(0, len(poscar_list), group_size)
                ]
                task_dirs = []
                for idx, _poscar_list in enumerate(grouped_poscar_list):
                    opt_path = Path(f"opt_path_{idx}")
                    task_dirs.append(work_dir / opt_path)
                    with set_directory(opt_path):
                        for poscar in _poscar_list:
                            Path(poscar.name).symlink_to(poscar)
                        Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
                        Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)
                task_names = [str(task_dir) for task_dir in task_dirs]
            else:
                temp_dir = work_dir / "opt_path_0"
                temp_dir.mkdir(parents=True, exist_ok=True)
                task_dirs = [temp_dir]
                task_names = [str(task_dir) for task_dir in task_dirs]

        return OPIO(
            {
                "task_names": task_names,
                "task_dirs": task_dirs,
                "caly_run_opt_file": work_dir / caly_run_opt_file.name,
                "caly_check_opt_file": work_dir / caly_check_opt_file.name,
            }
        )