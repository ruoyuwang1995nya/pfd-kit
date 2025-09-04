import json
import os
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from re import A
from typing import (
    Any,
    Dict,
    List,
    Set,
    Tuple,
    Union,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)


from ase import Atoms
from ase.io import read

class PrepFp(OP, ABC):
    r"""Prepares the working directories for first-principles (FP) tasks.

    A list of (same length as ip["confs"]) working directories
    containing all files needed to start FP tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "confs": Artifact(List[Path]),
                #"model_file": Artifact(Path, optional=True),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
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

            - `config` : (`dict`) Should have `config['inputs']`, which defines the input files of the FP task.
            - `confs` : (`Artifact(List[Path])`) Configurations for the FP tasks. Stored in folders as deepmd/npy format. Can be parsed as dpdata.MultiSystems.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the FP. The order fo the Paths should be consistent with `op["task_names"]`
        """

        config = ip["config"]
        confs = ip["confs"] 
        confs_ls=[]
        for cc in confs:
            confs_ls.extend(read(cc,index=":"))
        # loop over atoms
        task_names, task_paths = self._create_tasks(
            confs=confs_ls,
            config=config,
        )
        return OPIO(
            {
                "task_names": task_names,
                "task_paths": task_paths,
            }
        )
    @abstractmethod
    def _create_tasks(
        self,
        *args,
        **kwargs,
        )->Tuple[List[str],List[Path]]:
        pass
    
    
    