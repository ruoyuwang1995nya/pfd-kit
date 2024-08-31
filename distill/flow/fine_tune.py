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
    ShellOPTemplate
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

from dpgen2.op import (
        PrepLmp,
        PrepDPTrain,
        RunDPTrain
)
from dpgen2.superop import (
        PrepRunLmp,
        PrepRunDPTrain,
        PrepRunFp
    )
from regex import D
from distill.op import (
    RunLmp
)
from distill.superop import (
    ExplFinetuneLoop,
    ExplFinetuneBlock
)

class FineTune(Steps):
    def __init__(
        self,
        name:str,
        pert_gen_op: Type[OP],
        prep_run_fp_op: OPTemplate,
        collect_data_op:Type[OP],
        prep_run_dp_train_op: OPTemplate,
        md_expl_op: OPTemplate,
        expl_finetune_loop_op:OPTemplate,
        inference_op: Type[OP],
        pert_gen_step_config: dict,
        collect_data_step_config:dict,
        inference_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
        skip_aimd: bool = True
    ):
        self._input_parameters ={
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "config":InputParameter(), # Total input parameter file: to be changed in the future
            # md exploration 
            "numb_models": InputParameter(type=int),
            "expl_tasks": InputParameter(),
            "max_iter": InputParameter(),
            "explore_config": InputParameter(),
            # training
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "inference_config": InputParameter(),
            # fp calculation for labeling
            "fp_config":InputParameter(),
            # fp exploration
            "aimd_config": InputParameter(),
            "collect_data_config":InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "expl_models": InputArtifact(optional=True),
            "init_models": InputArtifact(),
            "iter_data": InputArtifact(optional=True),
            "init_data": InputArtifact(optional=True),
            "validation_data": InputArtifact(optional=True)
        }
        self._output_parameters ={
            
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
            pert_gen_op=pert_gen_op,
            prep_run_fp_op=prep_run_fp_op,
            prep_run_dp_train_op=prep_run_dp_train_op,
            collect_data_op=collect_data_op,
            inference_op=inference_op,
            md_expl_op=md_expl_op,
            expl_finetune_loop_op=expl_finetune_loop_op,
            pert_gen_step_config=pert_gen_step_config,
            collect_data_config=collect_data_step_config,
            inference_config=inference_step_config,
            upload_python_packages=upload_python_packages,
            skip_aimd=skip_aimd
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
    md_expl_op: OPTemplate,
    expl_finetune_loop_op: OPTemplate,
    collect_data_op: Type[OP],
    inference_op: Type[OP],
    pert_gen_step_config: dict,
    collect_data_config: dict,
    inference_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    skip_aimd: bool = True
):
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))
    
    collect_data_step_config = deepcopy(collect_data_config)
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
    
    if skip_aimd is True: 
        loop_iter_data=ft_steps.inputs.artifacts.get("iter_data")
    else:
        prep_run_fp = Step(
            name=name + "-prep-run-fp",
            template=prep_run_fp_op,
            parameters={
                "block_id": ft_steps.inputs.parameters["block_id"],
                "fp_config": ft_steps.inputs.parameters["aimd_config"],
                "type_map": ft_steps.inputs.parameters["type_map"],
        },
            artifacts={
                "confs": pert_gen.outputs.artifacts["confs"],
        },
            key="--".join(
                ["aimd-expl", "prep-run-fp"]
        ),
    )
        ft_steps.add(prep_run_fp)
    # Collect AIMD result
        collect_data = Step(
            name=name + "-collect-aimd",
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
            key="--".join(
                ["aimd-expl", "collect-data"]
            ),
            executor=collect_data_executor,
            **collect_data_step_config,
        )
        ft_steps.add(collect_data)
        loop_iter_data=collect_data.outputs.artifacts["multi_systems"]
        
        # train the model
        prep_run_ft = Step(
            name + "-prep-run-dp-train",
            template=prep_run_dp_train_op,
            parameters={
            "block_id": "aimd-init-ft",
            "train_config": ft_steps.inputs.parameters["train_config"],
            "numb_models": ft_steps.inputs.parameters["numb_models"],
            "template_script": ft_steps.inputs.parameters["template_script"],
            "run_optional_parameter":{
                "mixed_type": False,
                "finetune_mode": "finetune"}},
            artifacts={
            "init_models": ft_steps.inputs.artifacts["init_model"],
            "init_data": ft_steps.inputs.artifacts["init_data"],
            "iter_data": collect_data.outputs.artifacts["multi_systems"]#steps.inputs.artifacts["iter_data"],
        },
            key="--".join(
                ["aimd-init-ft", "prep-run-train"]))
        ft_steps.add(prep_run_ft)
        
    ############################
    # MD exploration and finetune
    #############################
    loop= Step(
        name = "ft-loop",
        template=expl_finetune_loop_op,
        parameters={
                "fp_config": ft_steps.inputs.parameters["fp_config"],
                "type_map": ft_steps.inputs.parameters["type_map"],
                "mass_map": ft_steps.inputs.parameters["mass_map"],
                "expl_tasks":ft_steps.inputs.parameters["expl_tasks"],
                "numb_models": ft_steps.inputs.parameters["numb_models"],
                "template_script": ft_steps.inputs.parameters["template_script"], 
                "train_config": ft_steps.inputs.parameters["train_config"],
                "explore_config": ft_steps.inputs.parameters["explore_config"],
                "dp_test_validation_config": {},
                "max_iter":ft_steps.inputs.parameters["max_iter"],
                },
            artifacts={
                "systems": pert_gen.outputs.artifacts["pert_sys"], # starting systems for model deviation
                "current_model" : ft_steps.inputs.artifacts["expl_models"],
                "init_model": ft_steps.inputs.artifacts["init_models"], # starting point for finetune
                "init_data": ft_steps.inputs.artifacts["init_data"], # initial data for model finetune
                "iter_data": loop_iter_data #ft_steps.inputs.artifacts["iter_data"],
                },
            key="--".join(
                ["%s" % "test", "-fp"]))
    ft_steps.add(loop)
    
    ft_steps.outputs.artifacts[
        "fine_tuned_model"
        ]._from = loop.outputs.artifacts["ft_model"][0]
    ft_steps.outputs.artifacts[
        "fine_tune_report"
        ]._from = loop.outputs.artifacts["iter_data"]
    

