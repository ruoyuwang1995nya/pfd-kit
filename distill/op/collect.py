from genericpath import isdir
import json
import pickle
import dpdata
import glob
import os
from pathlib import Path
import random
from pathlib import (
    Path,
)
from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter
)


class CollectData(OP):
    r"""Collect data for direct inference
    """
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "systems":Artifact(List[Path]),
                "additional_systems": Artifact(List[Path],optional=True),
                "type_map":Parameter(Union[List[str],None]),
                "optional_parameters":Parameter(Dict)
            }
        )
    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "test_systems": Artifact(List[Path],optional=True),
                
            }
        )
    
    @staticmethod
    def tasks(
        sys_path: Union[Path,str],
        fmt: Optional[str] = None,
        type_map: Optional[List[str]] = None,
        labeled_data: bool = False,
        **config
        ):
        print(fmt)
        if fmt is not None:
            fmt=fmt
        else:
            fmt = get_sys_fmt(sys_path)
        if labeled_data:
            sys=dpdata.LabeledSystem(sys_path,fmt=fmt,type_map=type_map,**config)
        else:
            sys=dpdata.System(sys_path,fmt=fmt,type_map=type_map,**config)
        return sys

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
        systems=ip["systems"]
        if additional_systems := ip.get("additional_systems"):
            systems.extend(additional_systems)
        type_map=ip["type_map"]
        optional_parameters=ip.get("optional_parameters",{})
        test_size=optional_parameters.pop("test_size",None)
        system_partition=optional_parameters.pop("system_partition",False)
        
        if system_partition is True and test_size:
            systems, test_systems,test_sys_idx =get_system_partition(systems,test_size=test_size)
            multi_sys=get_multi_sys(systems,type_map,**optional_parameters)
            test_sys=get_multi_sys(test_systems,type_map,**optional_parameters)    
            print("The following systems are selected as test systems: ", test_sys_idx)
        elif test_size:
            multi_sys=get_multi_sys(systems,type_map,**optional_parameters)
            multi_sys, test_sys, test_sys_idx = multi_sys.train_test_split(test_size=test_size,seed=1)  
            print(test_sys_idx)
        else:
            multi_sys=get_multi_sys(systems,type_map,**optional_parameters)
            test_sys=dpdata.MultiSystems()

        multi_sys.to('deepmd/npy','systems')
        test_sys.to('deepmd/npy','test_systems')
        return OPIO(
            {
                "systems":[path for path in Path("systems").iterdir() if path.is_dir()],
                "test_systems": [path for path in Path("test_systems").iterdir() if path.is_dir()] if Path("test_systems").is_dir() else []
            }
        )

def get_multi_sys(
    systems: List[Union[Path,str]],
    type_map:List[str],
    **kwargs
):
    multi_sys=dpdata.MultiSystems()
    for sys_path in systems:
        sys=CollectData.tasks(sys_path,type_map=type_map,**kwargs)   
        multi_sys.append(sys)
    return multi_sys
    


def get_system_partition(
    systems: List[Union[str,Path]],
    test_size: Union[str,float],
):
    num_sys=len(systems)
    ls=[ii for ii in range(num_sys)]
    random.shuffle(ls)
    num_test_sys=int(test_size*num_sys)
    systems_train=[systems[ii] for ii in ls[num_test_sys:]]
    systems_test=[systems[ii] for ii in ls[:num_test_sys]]
    test_sys_idx = ls[:num_test_sys]
    return systems_train, systems_test, test_sys_idx

def get_sys_fmt(
    sys_path:Union[Path,str]
    ):
    "return dpdata format"
    if isinstance(sys_path,str):
        sys_path=Path(sys_path)
    if  (sys_path / "type.raw").is_file():
        return "deepmd/npy"
        
    elif sys_path.is_file() and sys_path.suffix==".dump":
        return "lammps/dump"
    else:
        raise NotImplementedError("Unknown data format")

class CollectData_old(OP):
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
    
    @staticmethod
    def tasks(
        sys_path,
        fmt: Optional[str] = None,
        type_map: Optional[List[str]] = None,
        labeled_data: bool = False,
        **config
        ):
        if labeled_data:
            sys=dpdata.LabeledSystem(sys_path,fmt=fmt,type_map=type_map,**config)
        else:
            sys=dpdata.System(sys_path,fmt=fmt,type_map=type_map,**config)
        return sys

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
