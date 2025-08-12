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
    OutputParameter,
    Outputs,
    Step,
    Steps,
    if_expression,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)
from pfd.utils.step_config import init_executor
from pfd.op import EvalConv, ModelTestOP, StageScheduler


class ExplFinetuneBlock(Steps):
    def __init__(
        self,
        name: str,
        prep_run_explore_op: OPTemplate,
        prep_run_fp_op: OPTemplate,
        collect_data_op: Type[OP],
        select_confs_op: Type[OP],
        prep_run_train_op: OPTemplate,
        inference_op: Type[OP],
        collect_data_step_config: dict,
        inference_step_config: dict,
        select_confs_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "expl_tasks": InputParameter(),  # Total input parameter file: to be changed in the future
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "fp_config": InputParameter(),
            "collect_data_config": InputParameter(),
            "dp_test_validation_config": InputParameter(),
            "conf_selector": InputParameter(),
            "conf_filters_conv": InputParameter(value=None),
            "converge_config": InputParameter(value={}),
            "inference_config": InputParameter(),
            "finetune_mode": InputParameter(value="finetune"),
        }
        self._input_artifacts = {
            "systems": InputArtifact(),  # starting systems for model deviation
            "current_model": InputArtifact(),  # model for exploration
            "init_model": InputArtifact(),  # starting point for finetune
            "init_data": InputArtifact(),  # initial data for model finetune
            "iter_data": InputArtifact(
                optional=True
            ),  # datas collected during previous exploration
        }
        self._output_parameters = {
            "converged": OutputParameter(),
            "report": OutputParameter(default=None),
        }
        self._output_artifacts = {
            "ft_model": OutputArtifact(),
            "iter_data": OutputArtifact(),
            "test_report": OutputArtifact(),
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
        self = _expl_ft_blk(
            self,
            name=name,
            prep_run_explore_op=prep_run_explore_op,
            prep_run_fp_op=prep_run_fp_op,
            prep_run_train_op=prep_run_train_op,
            select_confs_op=select_confs_op,
            inference_op=inference_op,
            collect_data_op=collect_data_op,
            collect_data_step_config=collect_data_step_config,
            inference_step_config=inference_step_config,
            select_confs_step_config=select_confs_step_config,
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


class ExplFinetuneLoop(Steps):
    def __init__(
        self,
        name: str,
        expl_ft_blk_op: OPTemplate,
        scheduler_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(value="000"),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "conf_selector": InputParameter(),
            "conf_filters_conv": InputParameter(value=None),
            "fp_config": InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "dp_test_validation_config": InputParameter(),
            "converge_config": InputParameter(value={}),
            "scheduler": InputParameter(),
            "converged": InputParameter(value=False),
            "inference_config": InputParameter(),
            "report": InputParameter(value=None),
        }
        self._input_artifacts = {
            "systems": InputArtifact(),  # starting systems for model deviation
            "current_model": InputArtifact(),  # model for exploration
            "init_model": InputArtifact(),  # starting point for finetune
            "init_data": InputArtifact(),  # initial data for model finetune
            "iter_data": InputArtifact(
                optional=True
            ),  # datas collected during previous exploration
        }

        self._output_parameters = {"report": OutputParameter(default=None)}
        self._output_artifacts = {
            "ft_model": OutputArtifact(),
            "iter_data": OutputArtifact(),  # data collected after exploration
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
        self = _loop(
            self,
            name=name,
            expl_ft_blk_op=expl_ft_blk_op,
            scheduler_config=scheduler_config,
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
    name: str,
    prep_run_explore_op: OPTemplate,
    prep_run_fp_op: OPTemplate,
    prep_run_train_op: OPTemplate,
    select_confs_op: Type[OP],
    inference_op: Type[OP],
    collect_data_op: Type[OP],
    select_confs_step_config: dict,
    collect_data_step_config: dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    collect_data_step_config = deepcopy(collect_data_step_config)
    collect_data_template_config = collect_data_step_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))

    inference_step_config = deepcopy(inference_step_config)
    inference_template_config = inference_step_config.pop("template_config")
    inference_executor = init_executor(inference_step_config.pop("executor"))

    select_confs_step_config = deepcopy(select_confs_step_config)
    select_confs_template_config = select_confs_step_config.pop("template_config")
    select_confs_executor = init_executor(select_confs_step_config.pop("executor"))

    prep_run_explore = Step(
        name + "-prep-run-explore",
        template=prep_run_explore_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "explore_config": steps.inputs.parameters["explore_config"],
            "type_map": steps.inputs.parameters["type_map"],
            "expl_task_grp": steps.inputs.parameters["expl_tasks"],
        },
        artifacts={"models": steps.inputs.artifacts["current_model"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-explore"]),
    )
    steps.add(prep_run_explore)

    # select reasonable configurations
    select_confs = Step(
        name + "-select-confs",
        template=PythonOPTemplate(
            select_confs_op,
            output_artifact_archive={"confs": None},
            python_packages=upload_python_packages,
            **select_confs_template_config,
        ),
        parameters={
            "conf_selector": steps.inputs.parameters["conf_selector"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={"trajs": prep_run_explore.outputs.artifacts["trajs"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "select-confs"]),
        executor=select_confs_executor,
        **select_confs_step_config,
    )
    steps.add(select_confs)

    ## fp calculation
    prep_run_fp = Step(
        name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "fp_config": steps.inputs.parameters["fp_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": select_confs.outputs.artifacts["confs"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-fp"]),
    )
    steps.add(prep_run_fp)

    ## inference with expl_model
    dp_test = Step(
        name + "-test-model",
        template=PythonOPTemplate(
            inference_op,
            python_packages=upload_python_packages,
            **inference_template_config,
        ),
        parameters={
            "inference_config": steps.inputs.parameters["inference_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "systems": prep_run_fp.outputs.artifacts["labeled_data"],
            "model": steps.inputs.artifacts[
                "current_model"
            ],  # expl_model[0] #prep_run_ft.outputs.artifacts["models"][0]
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "test-model"]),
        executor=inference_executor,
    )
    steps.add(dp_test)

    evaluate = Step(
        name + "-check-converge",
        template=PythonOPTemplate(
            EvalConv,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "config": steps.inputs.parameters["converge_config"],
            "test_res": dp_test.outputs.parameters["test_res"],
            "conf_filters_conv": steps.inputs.parameters["conf_filters_conv"],
        },
        artifacts={"systems": prep_run_fp.outputs.artifacts["labeled_data"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "check-converge"]),
        **collect_data_step_config,
    )
    steps.add(evaluate)

    # add a step to
    collect_data = Step(
        name + "-collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "optional_parameters": steps.inputs.parameters["collect_data_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "systems": evaluate.outputs.artifacts["selected_systems"],
            "additional_multi_systems": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "collect-data"]),
        executor=collect_data_executor,
        **collect_data_step_config,
    )
    steps.add(collect_data)

    prep_run_ft = Step(
        name + "-prep-run-train",
        template=prep_run_train_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "train_config": steps.inputs.parameters["train_config"],
            "numb_models": steps.inputs.parameters["numb_models"],
            "template_script": steps.inputs.parameters["template_script"],
            "run_optional_parameter": {
                "mixed_type": False,
                "finetune_mode": steps.inputs.parameters["finetune_mode"],
            },
        },
        artifacts={
            "init_models": steps.inputs.artifacts["init_model"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": collect_data.outputs.artifacts[
                "multi_systems"
            ],  # steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-train"]),
        when="%s == false" % evaluate.outputs.parameters["converged"],
    )

    steps.add(prep_run_ft)
    steps.outputs.artifacts["ft_model"].from_expression = if_expression(
        _if=evaluate.outputs.parameters["converged"],
        _then=steps.inputs.artifacts["current_model"],
        _else=prep_run_ft.outputs.artifacts["models"],
    )

    steps.outputs.parameters[
        "converged"
    ].value_from_parameter = evaluate.outputs.parameters["converged"]
    steps.outputs.parameters[
        "report"
    ].value_from_parameter = evaluate.outputs.parameters["report"]
    steps.outputs.artifacts["iter_data"]._from = collect_data.outputs.artifacts[
        "multi_systems"
    ]
    steps.outputs.artifacts["test_report"]._from = dp_test.outputs.artifacts[
        "test_report"
    ]
    return steps


def _loop(
    loop,  # the loop Steps
    name: str,
    expl_ft_blk_op: OPTemplate,
    scheduler_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    scheduler_config = deepcopy(scheduler_config)
    scheduler_template_config = scheduler_config.pop("template_config")
    scheduler_executor = init_executor(scheduler_config.pop("executor"))

    # add a stage counter
    stage_scheduler = Step(
        name="loop-scheduler",
        template=PythonOPTemplate(
            StageScheduler,
            python_packages=upload_python_packages,
            **scheduler_template_config,
        ),
        parameters={
            "converged": loop.inputs.parameters["converged"],
            "scheduler": loop.inputs.parameters["scheduler"],
            "report": loop.inputs.parameters["report"],
        },
        artifacts={
            "systems": loop.inputs.artifacts["systems"],
            "init_model": loop.inputs.artifacts["init_model"],
            "current_model": loop.inputs.artifacts["current_model"],
        },
        key="--".join(["iter-%s" % loop.inputs.parameters["block_id"], "scheduler"]),
        executor=scheduler_executor,
        **scheduler_config,
    )
    loop.add(stage_scheduler)

    expl_ft_blk = Step(
        name=name + "-exploration-finetune",
        template=expl_ft_blk_op,
        parameters={
            "block_id": "iter-%s" % stage_scheduler.outputs.parameters["iter_id"],
            "type_map": loop.inputs.parameters["type_map"],
            "mass_map": loop.inputs.parameters["mass_map"],
            "expl_tasks": stage_scheduler.outputs.parameters["task_grp"],
            "converge_config": loop.inputs.parameters["converge_config"],
            "conf_selector": loop.inputs.parameters["conf_selector"],
            "conf_filters_conv": loop.inputs.parameters["conf_filters_conv"],
            "numb_models": loop.inputs.parameters["numb_models"],
            "template_script": loop.inputs.parameters["template_script"],
            "train_config": stage_scheduler.outputs.parameters["train_config"],
            "explore_config": loop.inputs.parameters["explore_config"],
            "inference_config": loop.inputs.parameters["inference_config"],
            "fp_config": loop.inputs.parameters["fp_config"],
            "collect_data_config": {
                "labeled_data": True,
                "multi_sys_name": stage_scheduler.outputs.parameters["iter_id"],
            },
            "dp_test_validation_config": loop.inputs.parameters[
                "dp_test_validation_config"
            ],
            "finetune_mode": stage_scheduler.outputs.parameters["finetune_mode"],
        },
        artifacts={
            "systems": loop.inputs.artifacts[
                "systems"
            ],  # starting systems for model deviation
            "current_model": loop.inputs.artifacts[
                "current_model"
            ],  # model for exploration
            "init_model": stage_scheduler.outputs.artifacts[
                "init_model_next"
            ],  # starting point for finetune
            "init_data": loop.inputs.artifacts["init_data"],
            "iter_data": loop.inputs.artifacts["iter_data"],
        },
        key="--".join(
            ["iter-%s" % stage_scheduler.outputs.parameters["iter_id"], "expl-ft-loop"]
        ),
        when="%s == false" % (stage_scheduler.outputs.parameters["converged"]),
    )
    loop.add(expl_ft_blk)

    # next iteration
    next_parameters = {
        "block_id": stage_scheduler.outputs.parameters["next_iter_id"],
        "type_map": loop.inputs.parameters["type_map"],
        "mass_map": loop.inputs.parameters["mass_map"],
        "conf_selector": loop.inputs.parameters["conf_selector"],
        "conf_filters_conv": loop.inputs.parameters["conf_filters_conv"],
        "converge_config": loop.inputs.parameters["converge_config"],
        "numb_models": loop.inputs.parameters["numb_models"],
        "template_script": loop.inputs.parameters["template_script"],
        "train_config": loop.inputs.parameters["train_config"],
        "explore_config": loop.inputs.parameters["explore_config"],
        "inference_config": loop.inputs.parameters["inference_config"],
        "fp_config": loop.inputs.parameters["fp_config"],
        "dp_test_validation_config": loop.inputs.parameters[
            "dp_test_validation_config"
        ],
        "scheduler": stage_scheduler.outputs.parameters["scheduler"],
        "converged": expl_ft_blk.outputs.parameters["converged"],
        "report": expl_ft_blk.outputs.parameters["report"],
    }
    next_step = Step(
        name=name + "-exploration-finetune-next",
        template=loop,
        parameters=next_parameters,
        artifacts={
            "systems": loop.inputs.artifacts["systems"],
            "init_model": loop.inputs.artifacts["init_model"],
            "current_model": expl_ft_blk.outputs.artifacts["ft_model"],
            "iter_data": expl_ft_blk.outputs.artifacts["iter_data"],
            "init_data": loop.inputs.artifacts["init_data"],
        },
        when="%s == false" % (stage_scheduler.outputs.parameters["converged"],),
        key="--".join(
            [
                "iter-%s" % stage_scheduler.outputs.parameters["next_iter_id"],
                "expl-ft-loop",
            ]
        ),
    )
    loop.add(next_step)
    # loop.outputs.parameters["report"].value_from_expression = if_expression(
    #    _if=stage_scheduler.outputs.parameters["converged"],
    #    _then=loop.inputs.parameters["report"],
    #    _else=next_step.outputs.parameters["report"],
    # )
    loop.outputs.artifacts["ft_model"].from_expression = if_expression(
        _if=stage_scheduler.outputs.parameters["converged"],
        _then=loop.inputs.artifacts["current_model"],
        _else=next_step.outputs.artifacts["ft_model"],
    )

    loop.outputs.artifacts["iter_data"].from_expression = if_expression(
        _if=stage_scheduler.outputs.parameters["converged"],
        _then=loop.inputs.artifacts["iter_data"],
        _else=next_step.outputs.artifacts["iter_data"],
    )
    return loop
