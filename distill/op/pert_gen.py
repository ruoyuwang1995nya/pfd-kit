import json
import pickle
import dpdata
import glob
import os
from pathlib import Path
from pathlib import (
    Path,
)
from typing import List

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter


class PertGen(OP):
    r"""Perturb the input configurations

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
                "init_confs": Artifact(List[Path]),
                "config": BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "pert_sys": Artifact(List[Path]),
                "confs": Artifact(List[Path]),  # multi system
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
            - `init_confs` : (`Artifact(List[Path])`) The paths to the input files of initial structure configurations.
            - `config` : (`BigParameter(dict)`) The input parameters for generating perturbed configurations.
        Returns
        -------
        op : dict
            Output dict with components:
            - `pert_sys`: (`Artifact(List[Path])`) The paths of perturbed systems. A list of `dpdata.System`.
            - `confs`: (`Artifact(List[Path])`) The prepared paths of perturbed systems. A list of `dpdata.MultiSystems`.
        """
        init_confs = ip["init_confs"]
        gen_config = ip["config"]  # ["conf_generation"]

        pert_configs = gen_config["pert_generation"]
        # default settings
        pert_ls = [0 for ii in range(len(init_confs))]

        for pert_idx in range(len(pert_configs)):
            if pert_configs[pert_idx]["conf_idx"] == "default":
                pert_ls = [pert_idx for ii in range(len(init_confs))]

            elif isinstance(pert_configs[pert_idx]["conf_idx"], list):
                for ii in pert_configs[pert_idx]["conf_idx"]:
                    pert_ls[ii] = pert_idx
            else:
                raise RuntimeError("Illegal specification of perturb generation")

        # get workdir
        wk_dir = Path("multi_sys")
        wk_dir.mkdir(exist_ok=True)
        multi_sys_ls = []
        sys_ls = []
        for ii in range(len(init_confs)):
            # create task directory
            name = "conf.%06d" % ii
            conf_path = Path(wk_dir) / name
            # os.chdir(conf_path)
            pert_param = pert_configs[pert_ls[ii]]
            cell_pert_fraction = pert_param.get("cell_pert_fraction", 0.05)
            pert_num = pert_param.get("pert_num", 200)
            atom_pert_distance = pert_param.get("atom_pert_distance", 0.2)
            atom_pert_style = pert_param.get("atom_pert_style", "normal")
            fmt = gen_config["init_configurations"]["fmt"]
            print(atom_pert_distance)
            orig_sys = dpdata.System(str(init_confs[ii]), fmt=fmt)
            pert_sys = orig_sys.perturb(
                pert_num=pert_num,
                cell_pert_fraction=cell_pert_fraction,
                atom_pert_distance=atom_pert_distance,
                atom_pert_style=atom_pert_style,
            )
            if_orig = pert_param.get("orig", False)
            if if_orig is True:
                pert_sys.append(orig_sys)
            pert_sys.to("deepmd/npy", str(conf_path))
            # os.chdir(wk_dir)
            sys_ls.append(conf_path)
            multi_sys_ls.append(wk_dir)
        return OPIO({"pert_sys": sys_ls, "confs": multi_sys_ls})
