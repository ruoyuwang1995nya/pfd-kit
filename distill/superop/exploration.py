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


class ExplorationBlock(Steps):
    def __init__(
        self,
        name: str,
        expl_tasks_op:Type[OP],
        prep_run_explore_op: OPTemplate,
        collect_data_op: Type[OP],
        gen_lmp_step_config:dict,
        collect_data_step_config:dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "config":InputParameter(), # Total input parameter file: to be changed in the future
            "explore_config": InputParameter(),
            "type_map":InputParameter(),
            
        }
        self._input_artifacts = {
            "systems": InputArtifact(),
            "models": InputArtifact(),
            "additional_systems": InputArtifact(optional=True),
        }
        self._output_parameters = {
            #"expl_infer_report": OutputParameter(),
        }
        self._output_artifacts = {
            "expl_systems": OutputArtifact()
            #"labeled_systems": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        # list of custom keys    
        self._my_keys = ["task-gen", "collect-data"]
        
        # key for the whole steps
        self._keys = (
            self._my_keys[:1]
            + prep_run_explore_op.keys
            + self._my_keys[1:2]
            #+ self._my_keys[2:3]
        )
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = "--".join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )

        self = _block_cl(
            self,
            self.step_keys,
            name,
            expl_tasks_op,
            prep_run_explore_op,
            collect_data_op,
            gen_lmp_step_config,
            collect_data_step_config,
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

    @property
    def keys(self):
        return self._keys

def _block_cl(
    block_steps:Steps,
    step_keys: Dict[str, Any],
    name:str,
    expl_tasks_op: Type[OP],
    prep_run_explore_op: OPTemplate,
    collect_data_op: Type[OP],
    gen_lmp_step_config: dict,
    collect_data_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
):
    ## get step config dict (not task config!)
    gen_lmp_step_config = deepcopy(gen_lmp_step_config)
    collect_data_step_config = deepcopy(collect_data_step_config)
    gen_lmp_template_config = gen_lmp_step_config.pop("template_config")
    collect_data_template_config = collect_data_step_config.pop("template_config")
    gen_lmp_executor = init_executor(gen_lmp_step_config.pop("executor"))
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))

    gen_lmp_tasks = Step(
        name + "gen-lmp-tasks",
        template=PythonOPTemplate(
            expl_tasks_op,
            python_packages=upload_python_packages,
            **gen_lmp_template_config
        ),
        parameters={
            "config": block_steps.inputs.parameters["config"],
        },
        artifacts={
            "systems": block_steps.inputs.artifacts["systems"],
        },
        key="--".join(
            ["%s" %block_steps.inputs.parameters["block_id"], "gen-lmp"]
        ),
        executor=gen_lmp_executor,
        **gen_lmp_step_config
        
    )
    block_steps.add(gen_lmp_tasks)
    
    prep_run_lmp = Step(
        name+"prep-run-lmp",
        template=prep_run_explore_op,
        parameters={
            "block_id": block_steps.inputs.parameters["block_id"],
            "explore_config":block_steps.inputs.parameters["explore_config"],
            "type_map": block_steps.inputs.parameters["type_map"],
            "expl_task_grp": gen_lmp_tasks.outputs.parameters["lmp_task_grp"],
        },
        artifacts={
            "models":block_steps.inputs.artifacts["models"]
        },
        key="--".join(
            ["%s" %block_steps.inputs.parameters["block_id"], "prep-run-lmp"]
        )
    )
    block_steps.add(prep_run_lmp)
    
    collect_data = Step(
        name+"collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config
        ),
        parameters={
            "type_map": block_steps.inputs.parameters["type_map"]
        },
        artifacts={
            "pert_sys": block_steps.inputs.artifacts["additional_systems"],
            "trajs":prep_run_lmp.outputs.artifacts["trajs"]
        },
        key="--".join(
            ["%s" %block_steps.inputs.parameters["block_id"], "collect-data"]
        ),
        executor=collect_data_executor,
        **collect_data_step_config
        
    )
    block_steps.add(collect_data)
    
    #inference = Step(
    #    "direct-inference",
    #    template=PythonOPTemplate(
    #        inference_op,
    #        python_packages=upload_python_packages,
    #        **infer_step_config
    #    ),
    #    parameters={
    #        "inference_config": block_steps.inputs.parameters["inference_config"]
    #    },
    #    artifacts={
    #        "systems": collect_data.outputs.artifacts["systems"],
    #        "model":block_steps.inputs.artifacts["models"][0]
    #    },
    #    key="--".join(
    #        ["%s" %block_steps.inputs.parameters["block_id"], "direct-inference"]
    #    )
    #)
    #block_steps.add(inference)    
    
    block_steps.outputs.artifacts["expl_systems"]._from = collect_data.outputs.artifacts["systems"]
    
    return block_steps