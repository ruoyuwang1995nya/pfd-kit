import json
import pickle
import dpdata
from pathlib import (
    Path,
)
from typing import (
    List,
    Dict,
    Tuple,
)
from dflow import upload_artifact


from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter

from dpdata.lammps.lmp import from_system_data

from dpgen2.constants import lmp_task_pattern, pytorch_model_name_pattern
from pdf.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
    NPTTaskGroup,
    make_lmp_task_group_from_config,
)


class TaskGen(OP):
    r"""Generate tasks for exploration

    A list of working directories (defined by `ip["task"]`)
    containing all files needed to start LAMMPS tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                # "lmp_task_grp": BigParameter(BaseExplorationTaskGroup),
                "systems": Artifact(List[Path]),
                "type_map": Parameter(List[str]),
                "mass_map": Parameter(List[float]),
                "expl_tasks": Parameter(
                    List[dict]
                ),  # this should be the generation config for each group
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "lmp_task_grp": BigParameter(BaseExplorationTaskGroup),
                # "type_map":Parameter(List[str])
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
        # perturbed systems
        systems = ip["systems"]
        expl_tasks = ip["expl_tasks"]
        type_map = ip["type_map"]
        mass_map = ip["mass_map"]
        expl_grp = BaseExplorationTaskGroup()
        for task in expl_tasks:
            print(task)
            task_grp = gen_expl_grp(systems, type_map, mass_map, task)
            for ii in task_grp.task_list:
                expl_grp.add_task(ii)
        return OPIO(
            {
                "lmp_task_grp": expl_grp,
            }
        )


def gen_expl_grp(systems, type_map: List, mass_map: List, task):
    task_grp = BaseExplorationTaskGroup()
    for idx in task["conf_idx"]:
        sys = dpdata.System(systems[idx], fmt="deepmd/npy", type_map=type_map)
        conf = [from_system_data(sys, f_idx=ii) for ii in range(sys.get_nframes())]
        task["exploration"]["model_name_pattern"] = pytorch_model_name_pattern
        task_grp_tmp = make_lmp_task_group_from_config(
            numb_models=1, mass_map=mass_map, config=task["exploration"].copy()
        )
        task_grp_tmp.set_conf(
            conf_list=conf,
            n_sample=task.get("n_sample", None),
            random_sample=True,
        )
        task_grp_tmp.make_task()
        for ii in task_grp_tmp.task_list:
            task_grp.add_task(ii)
    return task_grp
