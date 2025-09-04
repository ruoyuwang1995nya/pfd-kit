import json
import random
import sys
from abc import ABC, abstractmethod
from pathlib import (
    Path,
)
from typing import (
    Any,
    List,
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
from regex import sub

from pfd.constants import (
    train_script_name,
    train_task_pattern,
)


class PrepTrain(OP,ABC):
    r"""Prepares the working directories for DP training tasks.

    A list of (`numb_models`) working directories containing all files
    needed to start training tasks will be created. The paths of the
    directories will be returned as `op["task_paths"]`. The identities
    of the tasks are returned as `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "template_script": BigParameter(Union[dict, List[dict]]),
                "numb_models": int,
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
        numb_models = ip["numb_models"]
        osubdirs = []
        if type(template) != list:
            template = [template for ii in range(numb_models)]
        else:
            if not (len(template) == numb_models):
                raise RuntimeError(
                    f"length of the template list should be equal to {numb_models}"
                )

        workdir = Path(train_task_pattern)
        workdir.mkdir(exist_ok=True, parents=True)
        idict = self._process_script(template)
        
        # write input script
        fname = workdir / train_script_name
        with open(fname, "w") as fp:
            json.dump(idict, fp, indent=4)

        
        
        
        
        
        op = OPIO(
            {
                "task_names": osubdirs,
                "task_paths": [Path(ii) for ii in osubdirs],
            }
        )
        return op


    @abstractmethod
    def _process_script(
        self,
        input_dict,
    )-> Any:
        pass