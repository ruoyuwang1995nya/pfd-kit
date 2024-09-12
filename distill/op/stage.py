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
from traitlets import default

class StageScheduler(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'stages': Parameter(List[List[dict]]),
            'idx_stage':Parameter(int,default=0),
            #'converged':Parameter(bool,default=False)
        })
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'tasks': Parameter(List[dict]),
            #'idx_stage': Parameter(int),
            #'next_stage': Parameter(int)
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
        ) -> OPIO:
        return OPIO(
            {"tasks":ip["stages"][ip['idx_stage']]}
        )
