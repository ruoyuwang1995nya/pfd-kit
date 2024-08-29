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
    ShellOPTemplate,
    if_expression
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)


from dpgen2.utils.step_config import (
    init_executor
)

from dpgen2.superop import (
    PrepRunFp
)
from traitlets import Bool

from distill.op.collect import CollectData

class ExplFinetuneBlock(Steps):
    def __init__(
        self,
        name:str,
        md_expl_op: OPTemplate,
        prep_run_fp_op: OPTemplate,
        collect_data_op:Type[OP],
        prep_run_train_op: OPTemplate,
        inference_op: Type[OP],
        collect_data_step_config:dict,
        inference_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
        skip_training: bool = False 
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map":InputParameter(),
            "expl_tasks":InputParameter(), # Total input parameter file: to be changed in the future
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(), 
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "fp_config":InputParameter(),
            "collect_data_config":InputParameter(),
            "dp_test_validation_config": InputParameter(),
            
        }
        self._input_artifacts = {
            "systems": InputArtifact(), # starting systems for model deviation
            "current_model" : InputArtifact(), # model for exploration
            "init_model": InputArtifact(), # starting point for finetune
            "init_data": InputArtifact(), # initial data for model finetune
            "iter_data": InputArtifact(optional=True), # datas collected during previous exploration
        }
        self._output_parameters ={
        }
        self._output_artifacts = {
            "ft_model": OutputArtifact(),
            "iter_data": OutputArtifact(),
            "dp_test_report": OutputArtifact(),
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
        self=_expl_ft_blk(
            self,
            name,
            md_expl_op,
            prep_run_fp_op,
            prep_run_train_op,
            inference_op,
            collect_data_op,
            collect_data_step_config,
            inference_step_config,
            upload_python_packages=upload_python_packages,
            skip_training=skip_training
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


class ExplFinetuneLoop(Steps):
    def __init__(
        self,
        name:str,
        md_expl_op: OPTemplate,
        prep_run_fp_op: OPTemplate,
        collect_data_op:Type[OP],
        prep_run_train_op: OPTemplate,
        inference_op: Type[OP],
        collect_data_step_config:dict,
        inference_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
        
    ):        
        self._input_parameters = {
            "block_id": InputParameter(value=0),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            #"init_expl_model": InputParameter(type=bool),
            "expl_tasks":InputParameter(), # Total input parameter file: to be changed in the future
            "fp_config":InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(), 
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "dp_test_validation_config": InputParameter(),
            "max_iter": InputParameter(value=1)
        }
        self._input_artifacts = {
            "systems": InputArtifact(), # starting systems for model deviation
            "current_model" : InputArtifact(), # model for exploration
            "init_model": InputArtifact(), # starting point for finetune
            "init_data": InputArtifact(), # initial data for model finetune
            "iter_data": InputArtifact(optional=True), # datas collected during previous exploration
        }
        self._output_parameters ={
        }
        self._output_artifacts = {
            "ft_model": OutputArtifact(),
            #"dp_test_report": OutputArtifact(),
            #"dp_test_detail_files": OutputArtifact(),
            "iter_data": OutputArtifact() # data collected after exploration
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
        self=_loop(
            self,
            name=name,
            md_expl_op=md_expl_op,
            prep_run_fp_op=prep_run_fp_op,
            prep_run_train_op=prep_run_train_op,
            inference_op=inference_op,
            collect_data_op=collect_data_op,
            collect_data_step_config=collect_data_step_config,
            inference_step_config=inference_step_config,
            upload_python_packages=upload_python_packages,
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
    
def _expl_ft_blk(
    steps,
    name:str,
    md_expl_op: OPTemplate,
    prep_run_fp_op:OPTemplate,
    prep_run_train_op: OPTemplate,
    inference_op: Type[OP],
    collect_data_op: Type[OP],
    collect_data_step_config:dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    skip_training : bool =True
):
    collect_data_step_config = deepcopy(collect_data_step_config)
    collect_data_template_config = collect_data_step_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))
    #skip_training=steps.inputs.parameters["skip_training"]
    
    inference_step_config = deepcopy(inference_step_config)
    inference_template_config = inference_step_config.pop("template_config")
    inference_executor = init_executor(inference_step_config.pop("executor"))
    
    
    if skip_training is True:
        expl_model=steps.inputs.artifacts["current_model"]
    else:
        prep_run_ft = Step(
            name + "-prep-run-dp-train",
            template=prep_run_train_op,
            parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "train_config": steps.inputs.parameters["train_config"],
            "numb_models": steps.inputs.parameters["numb_models"],
            "template_script": steps.inputs.parameters["template_script"],
        },
            artifacts={
            "init_models": steps.inputs.artifacts["init_model"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
            key="--".join(
                ["%s" %steps.inputs.parameters["block_id"], "prep-run-train"]
            ),
        )
        steps.add(prep_run_ft)
        expl_model=steps.inputs.artifacts["current_model"]
        
    md_expl = Step(
        name+"-md-expl",
        template=md_expl_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "expl_tasks": steps.inputs.parameters["expl_tasks"],
            "explore_config": steps.inputs.parameters["explore_config"],
            "type_map": steps.inputs.parameters["type_map"],
            "mass_map": steps.inputs.parameters["mass_map"],
        },
        artifacts={
            "systems": steps.inputs.artifacts["systems"],
            "additional_systems" : steps.inputs.artifacts["systems"],
            "models": expl_model#steps.inputs.artifacts["current_model"], 
            },
        key="--".join(
                ["%s" %steps.inputs.parameters["block_id"], "md-expl"]
            ),
    )
    steps.add(md_expl)
    ## fp calculation
    prep_run_fp = Step(
        name=name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "fp_config": steps.inputs.parameters["fp_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": md_expl.outputs.artifacts["expl_systems"],
        },
        key="--".join(
            ["%s" % steps.inputs.parameters["block_id"], "prep-run-fp"]
        ),
    )
    steps.add(prep_run_fp)
    
    collect_data = Step(
        name=name + "-collect-data",
        template= PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "optional_parameters": steps.inputs.parameters["collect_data_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "systems": prep_run_fp.outputs.artifacts["labeled_data"],
            "additional_multi_systems": steps.inputs.artifacts["iter_data"]
        },
        key="--".join(
            ["%s" % steps.inputs.parameters["block_id"], "collect-data"]
        ),
        executor=collect_data_executor,
        **collect_data_step_config,
    )
    steps.add(collect_data)
    
    ## inference with expl_model
    dp_test = Step(
        name+ "-dp-test",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config
        ),
        parameters={
            "inference_config": {"task":"dp_test"},#ft_steps.inputs.parameters["inference_config"],
            "type_map": steps.inputs.parameters["type_map"]
        },
        artifacts={
            "systems": collect_data.outputs.artifacts["systems"],
            "model":expl_model[0] #prep_run_ft.outputs.artifacts["models"][0]
        },
        key="--".join(
            ["%s" %steps.inputs.parameters["block_id"], "validation-test"]
        ),
        executor=inference_executor
    )
    steps.add(dp_test)
    
    if skip_training is True:
        steps.outputs.artifacts["ft_model"]._from= steps.inputs.artifacts["init_model"]
    else:
        steps.outputs.artifacts["ft_model"]._from=prep_run_ft.outputs.artifacts["models"]
    steps.outputs.artifacts[
        "iter_data"
        ]._from = collect_data.outputs.artifacts["multi_systems"]
    steps.outputs.artifacts[
        "dp_test_report"
        ]._from = dp_test.outputs.artifacts["report"]
    return steps     
    
def _loop(
    loop, # the loop Steps
    name:str,
    md_expl_op: OPTemplate,
    prep_run_fp_op:OPTemplate,
    prep_run_train_op: OPTemplate,
    inference_op: Type[OP],
    collect_data_op: Type[OP],
    collect_data_step_config:dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
    # we just need a counter which supplies the iteration
    blk_counter_op = ShellOPTemplate(
            name='counter',
            image="alpine:3.15",
            script="echo 'This is iter {{inputs.parameters.blk_id}}' && "
                    "printf '%03d' $(({{inputs.parameters.blk_id}}+1)) 2>&1 | tee /tmp/blk_id.txt && "
                    "printf '%d' $(({{inputs.parameters.blk_id}}+1)) 2>&1 | tee /tmp/counter.txt"
                    )
    blk_counter_op.inputs.parameters = {"blk_id": InputParameter()}
    blk_counter_op.outputs.parameters = {
        "blk_id": OutputParameter(value_from_path="/tmp/blk_id.txt"),
        "counter": OutputParameter(value_from_path="/tmp/counter.txt")
        }
    blk_counter=Step(
        name="iter-counter", 
        template=blk_counter_op, 
        parameters={
                 "blk_id": loop.inputs.parameters["block_id"]},
        key="--".join(
                ["%s" % "iter-end","%s"%loop.inputs.parameters["block_id"] ]
                )
    )
    loop.add(blk_counter)
    expl_ft_blk_op=ExplFinetuneBlock(
            name="expl-ft",
            md_expl_op=md_expl_op,
            prep_run_fp_op=prep_run_fp_op,
            collect_data_op=collect_data_op,
            prep_run_train_op=prep_run_train_op,
            inference_op=inference_op,
            collect_data_step_config=collect_data_step_config,
            inference_step_config=inference_step_config,
            upload_python_packages=upload_python_packages,
            skip_training=False,
                )
    
    expl_ft_blk= Step(
        name=name+'-exploration-finetune',
        template=expl_ft_blk_op,
        parameters={
            "block_id": "iter-%s"% blk_counter.outputs.parameters["blk_id"],
            "type_map": loop.inputs.parameters["type_map"],
            "mass_map":loop.inputs.parameters["mass_map"],
            "expl_tasks":loop.inputs.parameters["expl_tasks"], # Total input parameter file: to be changed in the future
            "numb_models": loop.inputs.parameters["numb_models"],
            "template_script": loop.inputs.parameters["template_script"], 
            "train_config": loop.inputs.parameters["train_config"],
            "explore_config": loop.inputs.parameters["explore_config"],
            "fp_config":loop.inputs.parameters["fp_config"],
            "collect_data_config":{"labeled_data":True,
                                    "multi_sys_name": blk_counter.outputs.parameters["blk_id"]},
            "dp_test_validation_config": loop.inputs.parameters["dp_test_validation_config"]
            },
        artifacts={
            "systems": loop.inputs.artifacts["systems"], # starting systems for model deviation
            "current_model" : loop.inputs.artifacts["current_model"], # model for exploration
            "init_model": loop.inputs.artifacts["init_model"], # starting point for finetune
            "init_data": loop.inputs.artifacts["init_data"],
            "iter_data": loop.inputs.artifacts["iter_data"]
        },
        
        key="--".join(
                ["iter-%s"%blk_counter.outputs.parameters["blk_id"], "expl-ft-loop" ]
                )
        
    )
    loop.add(expl_ft_blk)
    
    # a evaluation step: to be added
    
    
    
    
    # next iteration
    next_parameters={
            "block_id": "iter-%s"% blk_counter.outputs.parameters["blk_id"],
            "type_map": loop.inputs.parameters["type_map"],
            "mass_map":loop.inputs.parameters["mass_map"],
            "expl_tasks":loop.inputs.parameters["expl_tasks"], # Total input parameter file: to be changed in the future
            "numb_models": loop.inputs.parameters["numb_models"],
            "template_script": loop.inputs.parameters["template_script"], 
            "train_config": loop.inputs.parameters["train_config"],
            "explore_config": loop.inputs.parameters["explore_config"],
            "fp_config":loop.inputs.parameters["fp_config"],
            "dp_test_validation_config": loop.inputs.parameters["dp_test_validation_config"]
            }
    next_step=Step(
        name=name+'-exploration-finetune-next',
        template=loop,
        parameters=next_parameters,
        artifacts={
            "systems":loop.inputs.artifacts["systems"],
            "init_model": loop.inputs.artifacts["init_model"],
            "current_model": expl_ft_blk.outputs.artifacts["ft_model"],
            "iter_data": expl_ft_blk.outputs.artifacts["iter_data"],
            "init_data": loop.inputs.artifacts["init_data"],
        },        
        when="%s < %s"%(
            blk_counter.outputs.parameters["counter"],
            loop.inputs.parameters["max_iter"]
        ),
        key="--".join(
                ["iter-%s" % blk_counter.outputs.parameters["blk_id"] ,"expl-ft-loop" ]
                )
    )
    loop.add(next_step)
    loop.outputs.artifacts["ft_model"
        ]._from = expl_ft_blk.outputs.artifacts["ft_model"]
    loop.outputs.artifacts[
        "iter_data"
        ]._from = expl_ft_blk.outputs.artifacts["iter_data"]
    return loop