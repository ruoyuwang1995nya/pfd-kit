import os
from copy import (
    deepcopy,
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
    Outputs,
    Step,
    Steps,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)


from pfd.utils.step_config import init_executor


class PFD(Steps):
    def __init__(
        self,
        name: str,
        prep_run_fp_op: OPTemplate,
        collect_data_op: Type[OP],
        train_op: Type[OP],
        expl_train_loop_op: OPTemplate,
        train_config: dict,
        collect_data_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
        init_train: bool =False,
        init_fp: bool = True,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            # md exploration
            #"explore_tasks": InputParameter(),
            "explore_config": InputParameter(),
            "conf_selector": InputParameter(),
            "select_confs_config": InputParameter(),
            "scheduler": InputParameter(),
            # training
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            # fp calculation for labeling
            "fp_config": InputParameter(),
            # fp exploration
            "collect_data_config": InputParameter(),
            "evaluate_config": InputParameter(),
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "expl_models": InputArtifact(optional=True),
            "init_models": InputArtifact(),
            #"iter_data": InputArtifact(optional=True),
            "init_data": InputArtifact(optional=True),
            "init_fp_confs": InputArtifact(optional=True),
            "validation_data": InputArtifact(optional=True),
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "model": OutputArtifact(),
            "iter_data": OutputArtifact(),
        }
        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters, artifacts=self._input_artifacts
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        self._keys = {
            "collect-data": "collect",
        }
        self = _pfd(
            self,
            name,
            prep_run_fp_op=prep_run_fp_op,
            train_op=train_op,
            train_config = train_config,
            collect_data_op=collect_data_op,
            expl_train_loop_op=expl_train_loop_op,
            collect_data_config=collect_data_step_config,
            upload_python_packages=upload_python_packages,
            init_train=init_train,
            init_fp=init_fp,
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


def _pfd(
    pfd_steps,
    name: str,
    prep_run_fp_op: OPTemplate,
    train_op: Type[OP],
    train_config: dict,
    expl_train_loop_op: OPTemplate,
    collect_data_op: Type[OP],
    collect_data_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    init_train: bool = False,
    init_fp: bool = False,
):
    train_config = deepcopy(train_config)
    train_template_config = train_config.pop("template_config")
    train_executor = init_executor(train_config.pop("executor"))

    collect_data_step_config = deepcopy(collect_data_config)
    collect_data_template_config = collect_data_step_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))

    if init_train is True:
        # if skip AIMD exploration
        if init_fp is False:
            init_data = pfd_steps.inputs.artifacts.get("init_data")
        # if execute AIMD exploration
        else:
            prep_run_fp = Step(
                "init-prep-run-fp",
                template=prep_run_fp_op,
                parameters={
                    "block_id": "init",
                    "fp_config": pfd_steps.inputs.parameters["aimd_config"],
                },
                artifacts={
                    "confs": pfd_steps.inputs.artifacts["init_fp_confs"],
                },
                key="--".join(["init", "prep-run-fp"]),
            )
            pfd_steps.add(prep_run_fp)
            collect_data = Step(
                "init-collect-data",
                template=PythonOPTemplate(
                    collect_data_op,
                    python_packages=upload_python_packages,
                    **collect_data_template_config
                ),
                parameters={
                    "optional_parameters": {
                        
                    },
                },
                artifacts={
                    "systems": prep_run_fp.outputs.artifacts["labeled_data"],
                    "pre_structures": pfd_steps.inputs.artifacts["init_data"],
                },
                key="--".join(["init", "collect-data"]),
                executor=collect_data_executor,
                **collect_data_step_config
            )
            pfd_steps.add(collect_data)
            init_data = collect_data.outputs.artifacts["all_structures"]

        # model training 
        prep_run_ft = Step(
            name + "-prep-run-train",
            template=PythonOPTemplate(
                train_op,
                python_packages=upload_python_packages,
                **train_template_config,
                ),
            parameters={
                "train_config": pfd_steps.inputs.parameters["train_config"],
                "template_script": pfd_steps.inputs.parameters["template_script"]
            },
            artifacts={
                "init_model": pfd_steps.inputs.artifacts["init_models"],
                "iter_data": init_data, # as iter_data is not optional
            },
            key="--".join(["init", "train"]),
            executor=train_executor,
            **train_config,
        )
        pfd_steps.add(prep_run_ft)
        expl_models = prep_run_ft.outputs.artifacts.get("models")
    # if skip initial model training
    else:
        #loop_iter_data = pfd_steps.inputs.artifacts.get("iter_data")
        expl_models = pfd_steps.inputs.artifacts.get("expl_models")

    loop = Step(
        name="expl-train-loop",
        template=expl_train_loop_op,
        parameters={
            "fp_config": pfd_steps.inputs.parameters["fp_config"],
            "conf_selector": pfd_steps.inputs.parameters["conf_selector"],
            "select_confs_config": pfd_steps.inputs.parameters["select_confs_config"],
            "template_script": pfd_steps.inputs.parameters["template_script"],
            "train_config": pfd_steps.inputs.parameters["train_config"],
            "explore_config": pfd_steps.inputs.parameters["explore_config"],
            #"explore_tasks": pfd_steps.inputs.parameters["explore_tasks"],
            "evaluate_config": pfd_steps.inputs.parameters["evaluate_config"],
            "scheduler": pfd_steps.inputs.parameters["scheduler"],
        },
        artifacts={
            "current_model": expl_models,  # pfd_steps.inputs.artifacts["expl_models"],
            "init_model": pfd_steps.inputs.artifacts["init_models"],  # starting point for finetune
            "init_data": init_data,  # initial data for model finetune
            #"iter_data": loop_iter_data,  # pfd_steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % "pfd", "loop"]),
    )
    pfd_steps.add(loop)
    pfd_steps.outputs.artifacts["model"]._from = loop.outputs.artifacts[
        "model"
    ]
    pfd_steps.outputs.artifacts["iter_data"]._from = loop.outputs.artifacts["iter_data"]
