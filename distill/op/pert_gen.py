import json
import pickle
import dpdata
import glob
import os
from pathlib import Path
from pathlib import (
    Path,
)
from typing import (
    List,
    Dict,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter
)


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
                #"lmp_task_grp": BigParameter(BaseExplorationTaskGroup),
                "init_confs":Artifact(List[Path]),
                "config":BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "pert_sys": Artifact(List[Path]),
                "confs": Artifact(List[Path]) # multi system
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
        init_confs=ip["init_confs"]
        gen_config=ip["config"]["conf_generation"]
        
        pert_configs=gen_config["pert_generation"]
        if pert_configs[0]["conf_idx"]=="default":
            pert_ls=[0 for ii in range(len(init_confs))]
        
        # get workdir
        wk_dir=os.getcwd()
        conf_paths=[]
        sys_paths=[]
        for ii in range(len(init_confs)):
            # create task directory
            name = "conf.%06d"%ii
            conf_path= Path(wk_dir) / name
            conf_path.mkdir(exist_ok=True)           
            # get pert config 
            os.chdir(conf_path)
            pert_param=pert_configs[pert_ls[ii]]
            cell_pert_fraction=pert_param.get("cell_pert_fraction",0.05)
            pert_num=pert_param.get("pert_num",200)
            atom_pert_distance=pert_param.get("atom_pert_distance",0.2)
            atom_pert_style=pert_param.get("atom_pert_style","normal")
            fmt=gen_config["init_configurations"]["fmt"]
            print(atom_pert_distance)
            orig_sys=dpdata.System(str(init_confs[ii]),fmt=fmt)
            pert_sys=orig_sys.perturb(
                    pert_num=pert_num,
                    cell_pert_fraction=cell_pert_fraction,
                    atom_pert_distance=atom_pert_distance,
                    atom_pert_style=atom_pert_style,
            )
            orig_sys.to("deepmd/npy","orig")
            pert_sys.to("deepmd/npy","pert")
            os.chdir(wk_dir)
            if_orig = pert_param.get("orig",False)
            if if_orig is True:
                sys_paths.append(conf_path/ "orig")
            sys_paths.append(conf_path / "pert")  
            conf_paths.append(conf_path)
        return OPIO(
            {
                "pert_sys":sys_paths,
                "confs":conf_paths
            }
        )
