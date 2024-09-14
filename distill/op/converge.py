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
from distill.exploration.converge import (
    ConvTypes
)


class EvalConv(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
            'config': Parameter(dict,default={}),
            'systems': Artifact(List[Path],optional=True),
            'test_res': BigParameter(List[Dict])})

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False)})

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        # implement 
        config=ip["config"]
        conv_type=config.pop("type")
        if conv_type in ConvTypes:
            conv=ConvTypes[conv_type]()
        else:
            raise NotImplementedError("%s is not implemented!"%conv_type)
        converged=conv.check_conv(
            ip["test_res"],
            config)
        return OPIO(
            {'converged':converged})
    
    
    
        
class NextLoop(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'converged': Parameter(bool, default=False),
            'iter_numb': Parameter(int,default=0),
            'max_iter': Parameter(int,default=1),
            "idx_stage": Parameter(int,default=0),
            "stages": Parameter(List[List[dict]]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'stage_converged': Parameter(bool, default=False),
            'converged': Parameter(bool, default=False),
            "idx_stage": Parameter(int)
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        numb_stages=len(ip["stages"])
        op={"converged":False,
            "stage_converged":False,
            "idx_stage":ip["idx_stage"]}
        if ip['iter_numb'] > ip['max_iter']:
            op['converged']=True
            op['stage_converged']=True
        elif ip['converged'] is True and ip["idx_stage"]+1 >= numb_stages:
            op['converged']=True
            op['stage_converged']=True
        elif ip['converged'] is True and ip["idx_stage"]+1 < numb_stages:
            op['stage_converged']=True
            op['idx_stage']=ip["idx_stage"]+1
        print(op)
        return OPIO(op)
        
        
class IterCounter(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'iter_numb': Parameter(int,default=0),
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