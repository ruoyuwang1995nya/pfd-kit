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

from dpgen2.fp import (
    PrepFpOpAbacus
)

from dpgen2.superop import (
    PrepRunFp
)


class FineTune(Steps):
    def __init__(
        self,
        name:str,
        pert_gen_op: Type[OP],
        prep_run_fp_op: OPTemplate,
        collect_data_op:Type[OP],
        prep_run_dp_op: OPTemplate,
        inference_op: Type[OP],
        #collect_data_config: dict, # for partition of traj
        pert_gen_step_config: dict,
        collect_data_step_config:dict,
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
            "inference_config": InputParameter(),
            "fp_config":InputParameter(),
            "collect_data_config":InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "init_models": InputArtifact(),
            "iter_data": InputArtifact(), # empty list
            "validation_data": InputArtifact(optional=True)
        }
        self._output_parameters ={
            #"fine_tune_report":OutputParameter()
        }
        self._output_artifacts = {
            "fine_tuned_model": OutputArtifact(),
            "fine_tune_report":OutputArtifact()
            
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
        self._keys={"collect-data":"collect",
                    }
        self = _fine_tune_cl(
            self,
            self._keys,
            name,
            pert_gen_op,
            prep_run_fp_op,
            prep_run_dp_op,
            collect_data_op,
            inference_op,
            #collect_data_config,
            pert_gen_step_config,
            collect_data_step_config,
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
        
def _fine_tune_cl(
    ft_steps,
    step_keys: Dict[str, Any],
    name:str,
    pert_gen_op: Type[OP],
    prep_run_fp_op: OPTemplate,
    prep_run_dp_train_op: OPTemplate,
    collect_data_op: Type[OP],
    inference_op: Type[OP],
    #collect_data_config:dict,
    pert_gen_step_config: dict,
    collect_data_step_config: dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
):
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))
    
    inference_step_config = deepcopy(inference_step_config)
    inference_template_config = inference_step_config.pop("template_config")
    inference_executor = init_executor(inference_step_config.pop("executor"))
    
    collect_data_step_config = deepcopy(collect_data_step_config)
    collect_data_template_config = collect_data_step_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))
    
    pert_gen = Step(
        name+"-pert-gen",
        template=PythonOPTemplate(
            pert_gen_op,
            python_packages=upload_python_packages,
            **pert_gen_template_config
        ),
        parameters={
            "config": ft_steps.inputs.parameters["config"]
        },
        artifacts={
            "init_confs":ft_steps.inputs.artifacts["init_confs"]
        },
        key="--".join(
            ["%s" %ft_steps.inputs.parameters["block_id"], "pert-gen"]
        ),
        executor=pert_gen_executor,
        **pert_gen_step_config
    )
    ft_steps.add(pert_gen)
    
    # AIMD simulation
    prep_run_fp = Step(
        name=name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": ft_steps.inputs.parameters["block_id"],
            "fp_config": ft_steps.inputs.parameters["fp_config"],
            "type_map": ft_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": pert_gen.outputs.artifacts["confs"],
        },
        key="--".join(
            ["%s" % ft_steps.inputs.parameters["block_id"], "prep-run-fp"]
        ),
    )
    ft_steps.add(prep_run_fp)
    
    
    # Collect AIMD result
    collect_data = Step(
        name=name + "-collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            #output_artifact_archive={"iter_data": None},
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "optional_parameters": ft_steps.inputs.parameters["collect_data_config"],
            "type_map": ft_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "systems": prep_run_fp.outputs.artifacts["labeled_data"],
        },
        key=step_keys["collect-data"],
        executor=collect_data_executor,
        **collect_data_step_config,
    )
    ft_steps.add(collect_data)
    
    # training set filtering
    check_force_train = Step(
        name+ "-data-train-check",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": ft_steps.inputs.parameters["inference_config"],
            "type_map": ft_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": collect_data.outputs.artifacts["systems"],
            "model":ft_steps.inputs.artifacts["init_models"][0]
        },
        key="--".join(
            ["%s" %ft_steps.inputs.parameters["block_id"], "validation"]
        ),
        executor=inference_executor
    )
    
    check_force_test = Step(
        name+ "-data-test-check",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": ft_steps.inputs.parameters["inference_config"],
            "type_map": ft_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": collect_data.outputs.artifacts["test_systems"],
            "model":ft_steps.inputs.artifacts["init_models"][0]
        },
        key="--".join(
            ["%s" %ft_steps.inputs.parameters["block_id"], "validation"]
        ),
        executor=inference_executor
    )
    ft_steps.add([check_force_train,check_force_test])
    
    
    # fine-tune
    finetune_optional_parameter = {
        "mixed_type": False,#config["inputs"]["mixed_type"],
        "finetune_mode": "finetune",
    }
    
    prep_run_ft = Step(
        name + "-prep-run-dp-train",
        template=prep_run_dp_train_op,
        parameters={
            "block_id": ft_steps.inputs.parameters["block_id"],
            "train_config": ft_steps.inputs.parameters["train_config"],
            "numb_models": ft_steps.inputs.parameters["numb_models"],
            "template_script": ft_steps.inputs.parameters["template_script"],
            "run_optional_parameter": finetune_optional_parameter
        },
        artifacts={
            "init_models": ft_steps.inputs.artifacts["init_models"],
            "init_data": collect_data.outputs.artifacts["systems"],
            "iter_data": ft_steps.inputs.artifacts["iter_data"],
            
        },
        key="--".join(
            ["%s" % ft_steps.inputs.parameters["block_id"], "prep-run-train"]
        ),
    )
    ft_steps.add(prep_run_ft)
    
    # validation
    validation = Step(
        name+ "-dp-test-validation",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": {"task":"dp_test"},#ft_steps.inputs.parameters["inference_config"],
            "type_map": ft_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": collect_data.outputs.artifacts["test_systems"],
            "model":prep_run_ft.outputs.artifacts["models"][0]
        },
        key="--".join(
            ["%s" %ft_steps.inputs.parameters["block_id"], "validation"]
        ),
        executor=inference_executor
    )
    
    # training set
    validation_training_set = Step(
        name+ "-dp-test-training-set",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": {"task":"dp_test"},   #ft_steps.inputs.parameters["inference_config"],
            "type_map": ft_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": collect_data.outputs.artifacts["systems"],
            "model":prep_run_ft.outputs.artifacts["models"][0]
        },
        key="--".join(
            ["%s" %ft_steps.inputs.parameters["block_id"], "validation"]
        ),
        executor=inference_executor
    )
    ft_steps.add([validation,validation_training_set])  
    
    ft_steps.outputs.artifacts[
        "fine_tuned_model"
        ]._from = prep_run_ft.outputs.artifacts["models"][0]
    ft_steps.outputs.artifacts[
        "fine_tune_report"
        ]._from = validation.outputs.artifacts["report"]
    

