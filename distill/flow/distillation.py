import dis
import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    Outputs,
    OPTemplate,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)

from distill.op import (
    ModelTestOP
)

from dpgen2.utils.step_config import (
    init_executor
)

class Distillation(Steps):
    def __init__(
        self,
        name:str,
        pert_gen_op: Type[OP],
        expl_dist_loop_op: OPTemplate,
        pert_gen_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None
    ):
        self._input_parameters ={
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            # pert_gen
            "pert_config":InputParameter(), 
            # exploration
            "expl_stages":InputParameter(),
            "numb_models": InputParameter(type=int),
            "max_iter": InputParameter(),
            "explore_config": InputParameter(),
            "converge_config":InputParameter(),
            "test_size":InputParameter(),
            # training
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "type_map_train":InputParameter(),
            # other configurations
            "inference_config": InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "teacher_model" : InputArtifact(),
            "init_data": InputArtifact(optional=True),
            "iter_data": InputArtifact(optional=True), # empty list
            "validation_data": InputArtifact(optional=True)
        }
        self._output_parameters ={
            #"dp_test":OutputParameter()
        }
        self._output_artifacts = {
            "dist_model": OutputArtifact(),
            #"dp_test_report": OutputArtifact(),
            #"dp_test_detail_files": OutputArtifact()
        }
        super().__init__(
            name=name,
            inputs=Inputs(
                parameters = self._input_parameters,
                artifacts = self._input_artifacts 
            ),
            outputs = Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            )
        )
        
        self = _dist_cl(
            self,
            name,
            pert_gen_op,
            expl_dist_loop_op,
            pert_gen_config,
            upload_python_packages=upload_python_packages
        )
        
    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts
    pass
        
        
def _dist_cl(
    steps,
    name:str,
    pert_gen_op: Type[OP],
    expl_dist_loop_op:OPTemplate,
    pert_gen_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
    ):
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))
    pert_gen = Step(
        name+"-pert-gen",
        template=PythonOPTemplate(
            pert_gen_op,
            python_packages=upload_python_packages,
            **pert_gen_template_config
        ),
        parameters={
            "config": steps.inputs.parameters["pert_config"]},
        artifacts={
            "init_confs":steps.inputs.artifacts["init_confs"]},
        key="--".join(
            ["%s" %steps.inputs.parameters["block_id"], "pert-gen"]),
        executor=pert_gen_executor,
        **pert_gen_step_config
    )
    steps.add(pert_gen)
    
    loop= Step(
        name = "ft-loop",
        template=expl_dist_loop_op,
        parameters={
                "type_map": steps.inputs.parameters["type_map"],
                "mass_map": steps.inputs.parameters["mass_map"],
                "expl_stages":steps.inputs.parameters["expl_stages"],
                "numb_models": steps.inputs.parameters["numb_models"],
                "template_script": steps.inputs.parameters["template_script"], 
                "train_config": steps.inputs.parameters["train_config"],
                "explore_config": steps.inputs.parameters["explore_config"],
                "max_iter":steps.inputs.parameters["max_iter"],
                "converge_config": steps.inputs.parameters["converge_config"],
                "inference_config": steps.inputs.parameters["inference_config"],
                "test_size": steps.inputs.parameters["test_size"],
                "type_map_train": steps.inputs.parameters["type_map_train"],
                },
            artifacts={
                "systems": pert_gen.outputs.artifacts["pert_sys"], # starting systems for model deviation
                "teacher_model" : steps.inputs.artifacts["teacher_model"],
                "init_data": steps.inputs.artifacts["init_data"], # initial data for model finetune
                "iter_data": steps.inputs.artifacts["iter_data"]
                },
            key="--".join(
                ["%s" % "test", "-loop"]))
    steps.add(loop)
    
    steps.outputs.artifacts[
        "dist_model"
        ]._from = loop.outputs.artifacts["dist_model"]
    return steps    