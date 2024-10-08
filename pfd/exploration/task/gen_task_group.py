import dpdata
from dpdata.lammps.lmp import from_system_data
from typing import List
from pfd.constants import lmp_task_pattern, pytorch_model_name_pattern
from pfd.exploration.task import (
    BaseExplorationTaskGroup,
    make_lmp_task_group_from_config,
)


def gen_expl_grp_lmps(
    systems,
    type_map: List,
    mass_map: List,
    task,
):
    task_grp = []
    for idx in task["conf_idx"]:
        sys = dpdata.System(systems[idx], fmt="deepmd/npy", type_map=type_map)
        conf = [from_system_data(sys, f_idx=ii) for ii in range(sys.get_nframes())]
        task["exploration"]["model_name_pattern"] = pytorch_model_name_pattern
        task_grp = make_lmp_task_group_from_config(
            numb_models=1, mass_map=mass_map, config=task["exploration"].copy()
        )
        task_grp.set_conf(
            conf_list=conf,
            n_sample=task.get("n_sample", None),
            random_sample=True,
        )
    return task_grp
