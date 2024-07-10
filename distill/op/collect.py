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
    Union
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter
)

from dpgen2.constants import (
    lmp_task_pattern,
)
from dpgen2.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)


class CollectData(OP):
    r"""Collect data for direct inference
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "pert_sys":Artifact(List[Path],optional=True),
                "trajs":Artifact(List[Path]),
                "type_map":Parameter(Union[List[str],None]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(Path),
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
        pert_sys=ip["pert_sys"]
        trajs=ip["trajs"]
        type_map=ip["type_map"]
        
        
        multi_sys=dpdata.MultiSystems()
        if pert_sys:
            for sys_path in pert_sys:
                try:
                    sys=dpdata.System(sys_path,"deepmd/npy",type_map=type_map)
                    if len(sys) > 0:
                        multi_sys.append(sys)
                except:
                    print("Something went wrong with", sys)
        if trajs:
            for traj_path in trajs:
                try:
                    traj_obj=read_trj(
                        traj_path,
                        fmt='lammps/dump',
                        type_map=type_map
                    )
                    multi_sys.append(traj_obj.data)
                except:
                    print("Error in reading trajectory file at %d"%traj_path)
            
        multi_sys.to("deepmd/npy","systems")
        return OPIO(
            {
                "systems":Path("systems")
            }
        )


class read_trj():
    def __init__(
        self,
        trj_file_path,
        #fmt:str='lammps/dump',
        **kwargs
        #config: dict = {}
        ):
        #trj_fmt=config.get("traj_fmt","lammps/dump")
        
        self._data=dpdata.System(trj_file_path,**kwargs)
        
    @property
    def data(self):
        return self._data
    
    def to_system(
        self,
        fmt:str="deepmd/npy",
        file_path:str="data",
        **kwargs
        ):
        self.data.to(fmt,file_path,**kwargs)       
