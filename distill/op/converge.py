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

class EvalConv(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
            'config': Parameter(dict,default={})
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        return OPIO(
            {
                'converged':ip['converged']
            }
        )
        
class NextLoop(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
            'iter_numb': Parameter(int,default=0),
            'max_iter': Parameter(int,default=1)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        if ip['converged'] is True or ip['iter_numb'] > ip['max_iter']:
            return OPIO(
                {'converged':True})
        else:
            return OPIO()
        
        
class IterCounter(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'iter_numb': Parameter(int,default=0)
        })
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'iter_numb': Parameter(int),
            'iter_id': Parameter(str),
            'next_iter_numb': Parameter(int),
            'next_iter_id': Parameter(str)
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        return OPIO({
            "iter_numb": ip["iter_numb"],
            "iter_id":"%03d"% ip["iter_numb"],
            "next_iter_numb":ip["iter_numb"]+1,
            "next_iter_id": "%03d"%(ip["iter_numb"]+1)
        })