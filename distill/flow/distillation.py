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


from dpgen2.utils.step_config import (
    init_executor
)



class Distillation(Steps):
    def __init__(
        self,
        name:str,
        pert_gen_op: Type[OP],
        expl_block_op: OPTemplate,
        inference_op: Type[OP],
        prep_run_dp_op: OPTemplate,
        pert_gen_step_config: dict,
        inference_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None
    ):
        self._input_parameters ={
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "config":InputParameter(), # Total input parameter file: to be changed in the future
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "inference_config": InputParameter(),
            "inference_validation_config": InputParameter(),
            "dp_test_validation_config": InputParameter(),
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "teacher_model" : InputArtifact(),
            "iter_data": InputArtifact(), # empty list
            "validation_data": InputArtifact(optional=True)
        }
        self._output_parameters ={
            #"dp_test":OutputParameter()
        }
        self._output_artifacts = {
            "distill_model": OutputArtifact(),
            "dp_test_report": OutputArtifact(),
            "dp_test_detail_files": OutputArtifact()
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
            expl_block_op,
            prep_run_dp_op,
            inference_op,
            pert_gen_step_config,
            inference_step_config,
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
    dist_steps,
    name:str,
    pert_gen_op: Type[OP],
    expl_block_op: OPTemplate,
    prep_run_dp_op: OPTemplate,
    inference_op: Type[OP],
    pert_gen_step_config: dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
    ):
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))
    
    inference_step_config = deepcopy(inference_step_config)
    inference_template_config = inference_step_config.pop("template_config")
    inference_executor = init_executor(inference_step_config.pop("executor"))
    
    pert_gen = Step(
        name+"-pert-gen",
        template=PythonOPTemplate(
            pert_gen_op,
            python_packages=upload_python_packages,
            **pert_gen_template_config
        ),
        parameters={
            "config": dist_steps.inputs.parameters["config"]
        },
        artifacts={
            "init_confs":dist_steps.inputs.artifacts["init_confs"]
        },
        key="--".join(
            ["%s" %dist_steps.inputs.parameters["block_id"], "pert-gen"]
        ),
        executor=pert_gen_executor,
        **pert_gen_step_config
    )
    dist_steps.add(pert_gen)
    
    expl_block =Step(
        name+"-expl-blk",
        template=expl_block_op,
        parameters={
            "block_id": dist_steps.inputs.parameters["block_id"],
            "config": dist_steps.inputs.parameters["config"],
            "explore_config": dist_steps.inputs.parameters["explore_config"],
            "type_map": dist_steps.inputs.parameters["type_map"], 
        },
        artifacts={
            "systems": pert_gen.outputs.artifacts["pert_sys"],
            "additional_systems" : pert_gen.outputs.artifacts["pert_sys"],
            "models": dist_steps.inputs.artifacts["teacher_model"], 
        },
        key="--".join(["%s" %dist_steps.inputs.parameters["block_id"], "expl-infer-block"])
    )
    dist_steps.add(expl_block)
    
    inference = Step(
        name+ "-inference",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": dist_steps.inputs.parameters["inference_config"],
            "type_map": dist_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": expl_block.outputs.artifacts["expl_systems"],
            "model":dist_steps.inputs.artifacts["teacher_model"][0]
        },
         key="--".join(
            ["%s" %dist_steps.inputs.parameters["block_id"], "expl-inference"]
        ),
         executor=inference_executor
    )
    dist_steps.add(inference)    
    
    
    prep_run_dp=Step(
        name+"-prep-run-dp",
        template=prep_run_dp_op,
        parameters={
            "block_id": dist_steps.inputs.parameters["block_id"],
            "train_config":dist_steps.inputs.parameters["train_config"],
            "numb_models":dist_steps.inputs.parameters["numb_models"],
            "template_script": dist_steps.inputs.parameters["template_script"]
        },
        artifacts={
            "init_data":inference.outputs.artifacts["labeled_systems"],
            "iter_data":dist_steps.inputs.artifacts["iter_data"]
        },
        key="--".join(["%s" %dist_steps.inputs.parameters["block_id"], "prep-run-dp"]),
    )
    dist_steps.add(prep_run_dp)
    
    validation_expl =Step(
        name+"-validation",
        template=expl_block_op,
        parameters={
            "block_id": dist_steps.inputs.parameters["block_id"]+"val",
            "config": dist_steps.inputs.parameters["config"],
            "explore_config": dist_steps.inputs.parameters["explore_config"],
            "type_map": dist_steps.inputs.parameters["type_map"], 
        },
        artifacts={
            "systems": pert_gen.outputs.artifacts["pert_sys"],
            "additional_systems" : dist_steps.inputs.artifacts["validation_data"],
            "models": prep_run_dp.outputs.artifacts["models"]
        },
        key="--".join(["%s"%dist_steps.inputs.parameters["block_id"], "validation-block"]),
        
    )
    dist_steps.add(validation_expl)
    
    inference_validation = Step(
        name+ "-inference-validation",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": dist_steps.inputs.parameters["inference_validation_config"],
            "type_map": dist_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": validation_expl.outputs.artifacts["expl_systems"],
            "model":prep_run_dp.outputs.artifacts["models"][0]
        },
        key="--".join(
            ["%s" %dist_steps.inputs.parameters["block_id"], "-inference-validation"]
        ),
        executor=inference_executor
    )
    dist_steps.add(inference_validation)  
    
    dp_test_validation = Step(
        name+ "-dp-test-validation",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": dist_steps.inputs.parameters["dp_test_validation_config"],
            "type_map": dist_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": inference_validation.outputs.artifacts["labeled_systems"],
            "model":dist_steps.inputs.artifacts["teacher_model"][0]
        },
        key="--".join(
            ["%s" %dist_steps.inputs.parameters["block_id"], "-dp-test-validation"]
        ),
        executor=inference_executor
    )
    dist_steps.add(dp_test_validation)  
    
    dist_steps.outputs.artifacts[
        "distill_model"
        ]._from = prep_run_dp.outputs.artifacts["models"][0]
    dist_steps.outputs.artifacts[
        "dp_test_report"
        ]._from = dp_test_validation.outputs.artifacts["report"]
    dist_steps.outputs.artifacts[
        "dp_test_detail_files"
        ]._from = dp_test_validation.outputs.artifacts["dp_test"]
    return dist_steps    