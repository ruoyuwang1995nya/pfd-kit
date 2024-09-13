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
                "additional_multi_systems": Artifact(List[Path],optional=True),
                "type_map":Parameter(Union[List[str],None]),
                "optional_parameters":Parameter(Dict)
            }
        )
    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "multi_systems": Artifact(List[Path]),
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
        # List of systems
        systems=ip["systems"]
        if additional_systems := ip.get("additional_systems"):
            systems.extend(additional_systems)
        # List of multi_systems
        if multi_system:= ip.get("additional_multi_systems"):
            print("Multi_sys is", multi_system)
        else:
            multi_system=[]
        # Collects various types of data     
        type_map=ip["type_map"]
        optional_parameters=ip.get("optional_parameters",{})
        test_size=optional_parameters.pop("test_size",None)
        system_partition=optional_parameters.pop("system_partition",False)
        multi_sys_name=optional_parameters.pop("multi_sys_name","multi_system")
        print(optional_parameters)
        print(multi_sys_name)
        
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

        multi_sys.to('deepmd/npy',multi_sys_name)
        test_multi_sys_name="%s_test"%multi_sys_name
        test_sys.to('deepmd/npy',test_multi_sys_name)
        #print([path for path in Path("systems").iterdir() if path.is_dir()])
        multi_system.append(Path(multi_sys_name))
        return OPIO(
            {
                "multi_systems": multi_system,
                "systems":[path for path in Path(multi_sys_name).iterdir() if path.is_dir()],
                "test_systems": [path for path in Path(test_multi_sys_name).iterdir() if path.is_dir()] if Path(test_multi_sys_name).is_dir() else []
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
    systems: List[Union[Path,str]],
    test_size: Union[str,float],
):
    num_sys=len(systems)
    ls=[ii for ii in range(num_sys)]
    random.shuffle(ls)
    num_test_sys=int(test_size*num_sys)
    systems_train=[systems[ii] for ii in ls[num_test_sys:]]
    systems_test=[systems[ii] for ii in ls[:num_test_sys]]
    test_sys_idx = [str(systems[ii]) for ii in ls[:num_test_sys]]
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
    
