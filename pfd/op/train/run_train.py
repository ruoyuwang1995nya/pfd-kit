import glob
import json
import logging
from math import log
import os
import shutil
from pathlib import (
    Path,
)
from turtle import st
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from abc import ABC, abstractmethod

from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    NestedDict,
    OPIOSign,
    Parameter,
    TransientError,
)
from torch import log_

from pfd import op
from pfd.constants import (
    train_script_name,
    train_task_pattern,
)
from pfd.utils.chdir import (
    set_directory,
)
from pfd.utils.run_command import (
    run_command,
)

class RunTrain(OP,ABC):
    r"""Execute a DP training task. Train and freeze a DP model.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The
    DeePMD-kit training and freezing commands are exectuted from
    directory `task_name`.

    """

    default_optional_parameter = {}
    train_script_name = "input.json"
    model_file = "model.pb"
    log_file = "train.log"

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": dict,
                "task_name": BigParameter(str),
                "optional_parameter": Parameter(
                    dict,
                    default={}
                ),
                "task_path": Artifact(Path),
                "init_model": Artifact(Path, optional=True),
                "init_data": Artifact(NestedDict[Path],optional=True),
                "iter_data": Artifact(List[Path]),
                "valid_data": Artifact(NestedDict[Path], optional=True),
                "optional_files": Artifact(List[Path], optional=True),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "script": Artifact(Path),
                "model": Artifact(Path),
                "lcurve": Artifact(Path),
                "log": Artifact(Path),
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

            - `config`: (`dict`) The config of training task. Check `RunDPTrain.training_args` for definitions.
            - `task_name`: (`str`) The name of training task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepDPTrain`.
            - `init_model`: (`Artifact(Path)`) A frozen model to initialize the training.
            - `init_data`: (`Artifact(NestedDict[Path])`) Initial training data.
            - `iter_data`: (`Artifact(List[Path])`) Training data generated in the DPGEN iterations.

        Returns
        -------
        Any
            Output dict with components:
            - `script`: (`Artifact(Path)`) The training script.
            - `model`: (`Artifact(Path)`) The trained frozen model.
            - `lcurve`: (`Artifact(Path)`) The learning curve file.
            - `log`: (`Artifact(Path)`) The log file of training.

        Raises
        ------
        FatalError
            On the failure of training or freezing. Human intervention needed.
        """
        
        config = ip["config"] if ip["config"] is not None else {}
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        init_model = ip["init_model"]
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]
        valid_data = ip["valid_data"]
        optional_param = ip["optional_parameter"]#.get("finetune_mode", self.default_optional_parameter["finetune_mode"])
        
        # set working directory
        work_dir = Path(task_name)
        with set_directory(work_dir):
            # prepare training directory
            self.prepare_train(
                work_dir=work_dir,
                task_path=task_path,
                config=config,
                init_model=init_model,
                init_data=init_data,
                iter_data=iter_data,
                valid_data=valid_data,
                optional_param=optional_param,
                )
            
            # run training
            ret, out, err =  self.run_train()
            
            # write and print log
            self.write_log(ret=ret, out=out, err=err)
            
        return OPIO(
            {
                "script": work_dir / self.train_script_name,
                "model": work_dir / self.model_file,
                "lcurve": work_dir / "lcurve.out",
                "log": work_dir / self.log_file,
            }
        )
        
    @abstractmethod
    def prepare_train(
        self,
        *args,
        **kwargs
        ):
        pass

    @abstractmethod
    def run_train(
        self,
        *args, 
        **kwargs
        )-> Tuple[int,str,str]:
        pass
    
    @abstractmethod
    def write_log(
        self,
        *args,
        **kwargs
        ):
        pass
    

    @classmethod
    @abstractmethod
    def training_args(cls):
        return []

    @classmethod
    @abstractmethod
    def normalize_config(cls, data={}):
        ta = cls.training_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data
