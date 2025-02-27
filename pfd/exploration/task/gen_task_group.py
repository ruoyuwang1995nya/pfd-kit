import dpdata
from dpdata.lammps.lmp import from_system_data
from typing import List
from pathlib import Path
import copy
from pfd.constants import lmp_task_pattern, pytorch_model_name_pattern
from pfd.exploration.task import (
    BaseExplorationTaskGroup,
    make_lmp_task_group_from_config,
    normalize_lmp_task_group_config,
    normalize_caly_task_group_config,
    caly_normalize,
)

from pfd.exploration.task.caly_task_group import CalyTaskGroup

from dflow import ArgoStep, Step, Steps, Workflow, upload_artifact


def gen_expl_grp_lmps(
    sys,
    type_map: List,
    mass_map: List,
    task,
):
    sys = dpdata.System(sys, fmt="deepmd/npy", type_map=type_map)
    conf = [from_system_data(sys, f_idx=ii) for ii in range(sys.get_nframes())]
    # task["model_name_pattern"] = pytorch_model_name_pattern
    task_grp = make_lmp_task_group_from_config(
        numb_models=1, mass_map=mass_map, config=task
    )
    task_grp.set_conf(
        conf_list=conf, n_sample=task.get("n_sample", None), random_sample=True
    )
    return task_grp


def expl_grp_args_lmps(config):
    config = normalize_lmp_task_group_config(config, strict=True)
    # big_parameter=copy.deepcopy(config)
    if config["type"] == "lmp-template":
        lmp_template_fname = config.pop("lmp_template_fname")
        config["lmp_template"] = Path(lmp_template_fname).read_text().split("\n")
        if plm_template_fname := config.pop("plm_template_fname"):
            config["plm_template"] = Path(plm_template_fname).read_text().split("\n")
    return config


def expl_grp_args_caly(config):
    """Normalize input parameters of calypso exploration

    Args:
        config (dict): input config

    Returns:
        dict: normalized config
    """
    config.pop("type", None)
    config = normalize_caly_task_group_config(config, strict=True)
    return config


def gen_expl_grp_caly(
    task,
):
    tgroup = CalyTaskGroup()
    tgroup.set_params(**task)
    return tgroup
