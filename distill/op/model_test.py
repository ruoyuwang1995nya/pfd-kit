from pathlib import Path
import json
from typing import (
    List,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter
)
from distill.exploration.inference import ModelTypes
class ModelTestOP(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
                "systems":Artifact(List[Path]),
                "model":Artifact(Path),
                "type_map":Parameter(List),
                "inference_config":BigParameter(dict)})

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
                "test_report":Artifact(Path),
                "test_res":BigParameter(List[dict]),
                "test_res_dir":Artifact(Path)})
        
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        systems=ip["systems"]
        model_path=ip["model"]
        type_map=ip["type_map"]
        config=ip["inference_config"]
        
        model_type=config["model"]
        if model_type in ModelTypes.keys():
            Eval=ModelTypes[model_type]
        else:
            raise NotImplementedError(
                "%s is not implemented!"%model_type)
        res_total=[]
        report={}
        res_dir=Path("result")
        res_dir.mkdir(exist_ok=True)
        for sys in systems:
            name=sys.name
            evaluator=Eval(
                model=model_path,
                data=sys,
                type_map=type_map
                )
            res,rep=evaluator.evaluate(name)
            res_total.append(res)
            report[name]=rep
        with open('report.json','w') as fp:
            json.dump(report,fp,indent=4)
        return OPIO({
            "test_res":res_total,
            "test_report":Path('report.json'),
            "test_res_dir":res_dir
        })
        