import json
from abc import ABC, abstractmethod
from pathlib import (
    Path,
)
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Dict
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    Parameter,
    BigParameter,
    OPIOSign,
    NestedDict
)

from dargs import (
    Argument
)

from pfd.constants import (
    train_script_name,
    train_task_pattern,
)

from pfd.utils.chdir import (
    set_directory,
)

class Train(OP,ABC):
    r"""Prepares and run model training.

    A list of (`numb_models`) working directories containing all files
    needed to start training tasks will be created. The paths of the
    directories will be returned as `op["task_paths"]`. The identities
    of the tasks are returned as `op["task_names"]`.

    """
    
    default_optional_parameter = {}
    train_script_name = "input.json"
    model_name = "model.pb"
    log_file = "train.log"
    lcurve_file = "lcurve.out"

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
            #"block_id": Parameter(type=str, default=""),
            "train_config": Parameter(Dict,default={}),
            "run_optional_parameter": Parameter(Dict, default={}),
            "template_script": BigParameter(Union[dict, List[dict]]),

            "init_model": Artifact(Path, optional=True),
            "init_data": Artifact(List[Path], optional=True),
            "iter_data": Artifact(Path),
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

            - `template_script`: (`str` or `List[str]`) A template of the training script. Can be a `str` or `List[str]`. In the case of `str`, all training tasks share the same training input template, the only difference is the random number used to initialize the network parameters. In the case of `List[str]`, one training task uses one template from the list. The random numbers used to initialize the network parameters are differnt. The length of the list should be the same as `numb_models`.
            - `numb_models`: (`int`) Number of DP models to train.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. The order fo the Paths should be consistent with `op["task_names"]`

        """
        template = ip["template_script"]
        config = ip["train_config"]
        init_model = ip.get("init_model")
        init_data = ip.get("init_data")
        iter_data = ip["iter_data"]
        valid_data = ip.get("valid_data")
        optional_param = ip.get("run_optional_parameter", self.default_optional_parameter)

        work_dir = Path(train_task_pattern%0)
        work_dir.mkdir(exist_ok=True, parents=True)
        idict = self._process_script(template)
        
        # write input script
        fname = work_dir / train_script_name
        with open(fname, "w") as fp:
            json.dump(idict, fp, indent=4)

        
        with set_directory(work_dir):
            # prepare training directory
            self.prepare_train(
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
        
        
        op = OPIO(
            {
                "script": work_dir / self.train_script_name,
                "model": work_dir / self.model_name,
                "lcurve": work_dir / self.lcurve_file,
                "log": work_dir / self.log_file,
            }
        )
        return op


    @abstractmethod
    def _process_script(
        self,
        input_dict,
    )-> Any:
        pass
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
