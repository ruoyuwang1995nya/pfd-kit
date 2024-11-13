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
    if_expression,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)

from dpgen2.utils.step_config import init_executor
from pfd.op import ModelTestOP, EvalConv, IterCounter, StageScheduler, NextLoop


class ExplDistBlock(Steps):
    def __init__(
        self,
        name: str,
        prep_run_explore_op: OPTemplate,
        prep_run_train_op: OPTemplate,
        collect_data_op: Type[OP],
        inference_op: Type[OP],
        inference_config: dict,
        model_test_config: dict,
        collect_data_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "expl_tasks": InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "inference_config": InputParameter(),
            "converge_config": InputParameter(value={}),
            "conf_filters_conv": InputParameter(),
            "collect_data_config": InputParameter(value={}),
            "type_map_train": InputParameter(),
        }
        self._input_artifacts = {
            "systems": InputArtifact(),
            "teacher_model": InputArtifact(),
            "init_data": InputArtifact(optional=True),
            "iter_data": InputArtifact(optional=True),  # empty list
            "validation_data": InputArtifact(optional=True),
        }
        self._output_parameters = {"converged": OutputParameter()}
        self._output_artifacts = {
            "iter_data": OutputArtifact(),
            "dist_model": OutputArtifact(),
            "dp_test_report": OutputArtifact(),
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

        self = _expl_dist_cl(
            self,
            name,
            prep_run_explore_op,
            prep_run_train_op,
            collect_data_op,
            inference_op,
            inference_config,
            model_test_config,
            collect_data_config,
            upload_python_packages,
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


class ExplDistLoop(Steps):
    def __init__(
        self,
        name: str,
        expl_dist_blk_op: OPTemplate,
        scheduler_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(value="000"),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "conf_selector": InputParameter(value={}),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "converge_config": InputParameter(value={}),
            "conf_filters_conv": InputParameter(),
            "inference_config": InputParameter(),
            "test_size": InputParameter(value=0.1),
            "type_map_train": InputParameter(),
            "scheduler": InputParameter(),
            "converged": InputParameter(value=False),
        }
        self._input_artifacts = {
            "systems": InputArtifact(),  # starting systems for model deviation
            "teacher_model": InputArtifact(),  # model for exploration
            "init_data": InputArtifact(),  # initial data for model finetune
            "iter_data": InputArtifact(
                optional=True
            ),  # datas collected during previous exploration
        }
        self._output_parameters = {}

        self._output_artifacts = {
            "dist_model": OutputArtifact(),
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
        self = _loop(
            self,
            name=name,
            expl_dist_blk_op=expl_dist_blk_op,
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


def _expl_dist_cl(
    steps,
    name: str,
    prep_run_explore_op: OPTemplate,
    prep_run_train_op: OPTemplate,
    collect_data_op: Type[OP],
    inference_op: Type[OP],
    inference_config: dict,
    model_test_config: dict,
    collect_data_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):

    inference_config = deepcopy(inference_config)
    inference_template_config = inference_config.pop("template_config")
    inference_executor = init_executor(inference_config.pop("executor"))

    model_test_config = deepcopy(model_test_config)
    model_test_template_config = model_test_config.pop("template_config")
    model_test_executor = init_executor(model_test_config.pop("executor"))

    # essentially for utillity
    collect_data_config = deepcopy(collect_data_config)
    collect_data_template_config = collect_data_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_config.pop("executor"))

    prep_run_explore = Step(
        name + "prep-run-explore",
        template=prep_run_explore_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "explore_config": steps.inputs.parameters["explore_config"],
            "type_map": steps.inputs.parameters["type_map"],
            "expl_task_grp": steps.inputs.parameters["expl_tasks"],
        },
        artifacts={"models": steps.inputs.artifacts["teacher_model"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-explore"]),
    )
    steps.add(prep_run_explore)

    collect_data = Step(
        name + "collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "type_map": steps.inputs.parameters["type_map"],
            "optional_parameters": {"labeled_data": False}
            # steps.inputs.parameters["collect_data_config"],
        },
        artifacts={
            "systems": prep_run_explore.outputs.artifacts["trajs"],
            # "additional_multi_systems": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "collect-data"]),
        executor=collect_data_executor,
        **collect_data_config,
    )
    steps.add(collect_data)

    inference_train = Step(
        name + "-inference-train",
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
            "systems": collect_data.outputs.artifacts["systems"],
            "model": steps.inputs.artifacts["teacher_model"][0],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "inference-train"]),
        executor=inference_executor,
    )

    steps.add(inference_train)

    collect_data_train = Step(
        name + "collect-data-train",
        template=PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "type_map": steps.inputs.parameters["type_map"],
            "optional_parameters": steps.inputs.parameters["collect_data_config"],
        },
        artifacts={
            "systems": inference_train.outputs.artifacts["labeled_systems"],
            "additional_multi_systems": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(
            ["%s" % steps.inputs.parameters["block_id"], "collect-data-train"]
        ),
        executor=collect_data_executor,
        **collect_data_config,
    )
    steps.add(collect_data_train)

    prep_run_dp = Step(
        name + "-prep-run-dp",
        template=prep_run_train_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "train_config": steps.inputs.parameters["train_config"],
            "numb_models": steps.inputs.parameters["numb_models"],
            "template_script": steps.inputs.parameters["template_script"],
        },
        artifacts={
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": collect_data_train.outputs.artifacts["multi_systems"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-train"]),
    )
    steps.add(prep_run_dp)

    # the exploration steps for validation
    dp_test = Step(
        name + "-dp-test",
        template=PythonOPTemplate(
            ModelTestOP,
            python_packages=upload_python_packages,
            **model_test_template_config,
        ),
        parameters={
            "inference_config": {
                "model": "dp"
            },  # ft_steps.inputs.parameters["inference_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "systems": collect_data_train.outputs.artifacts["test_systems"],
            "model": prep_run_dp.outputs.artifacts["models"][0],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "validation-test"]),
        executor=model_test_executor,
    )
    steps.add(dp_test)

    evaluate = Step(
        name="evaluate-converge",
        template=PythonOPTemplate(
            EvalConv,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "config": steps.inputs.parameters["converge_config"],
            "test_res": dp_test.outputs.parameters["test_res"],
        },
        artifacts={"systems": collect_data_train.outputs.artifacts["test_systems"]},
        key="--".join(
            ["%s" % steps.inputs.parameters["block_id"], "evaluate-converge"]
        ),
        **collect_data_config,
    )
    steps.add(evaluate)
    steps.outputs.artifacts["iter_data"]._from = collect_data_train.outputs.artifacts[
        "multi_systems"
    ]
    steps.outputs.artifacts["dist_model"]._from = prep_run_dp.outputs.artifacts[
        "models"
    ][0]
    steps.outputs.artifacts["dp_test_report"]._from = dp_test.outputs.artifacts[
        "test_report"
    ]
    steps.outputs.parameters[
        "converged"
    ].value_from_parameter = evaluate.outputs.parameters["converged"]
    return steps


def _loop(
    loop,  # the loop Steps
    name: str,
    expl_dist_blk_op: OPTemplate,
    scheduler_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):

    scheduler_config = deepcopy(scheduler_config)
    scheduler_template_config = scheduler_config.pop("template_config")
    scheduler_executor = init_executor(scheduler_config.pop("executor"))

    # add a stage counter
    stage_scheduler = Step(
        name="stage-scheduler",
        template=PythonOPTemplate(
            StageScheduler,
            python_packages=upload_python_packages,
            **scheduler_template_config,
        ),
        parameters={
            "converged": loop.inputs.parameters["converged"],
            "scheduler": loop.inputs.parameters["scheduler"],
        },
        artifacts={
            "systems": loop.inputs.artifacts["systems"],
        },
        key="--".join(["iter-%s" % loop.inputs.parameters["block_id"], "scheduler"]),
        executor=scheduler_executor,
        **scheduler_config,
    )
    loop.add(stage_scheduler)

    expl_dist_blk = Step(
        name=name + "-exploration-dist",
        template=expl_dist_blk_op,
        parameters={
            "block_id": "iter-%s" % stage_scheduler.outputs.parameters["iter_id"],
            "type_map": loop.inputs.parameters["type_map"],
            "mass_map": loop.inputs.parameters["mass_map"],
            "expl_tasks": stage_scheduler.outputs.parameters["task_grp"],
            "converge_config": loop.inputs.parameters["converge_config"],
            "conf_filters_conv": loop.inputs.parameters["conf_filters_conv"],
            "numb_models": loop.inputs.parameters["numb_models"],
            "template_script": loop.inputs.parameters["template_script"],
            "train_config": loop.inputs.parameters["train_config"],
            "explore_config": loop.inputs.parameters["explore_config"],
            "inference_config": loop.inputs.parameters["inference_config"],
            "type_map_train": loop.inputs.parameters["type_map_train"],
            "collect_data_config": {
                "labeled_data": True,
                "test_size": loop.inputs.parameters["test_size"],
                "multi_sys_name": stage_scheduler.outputs.parameters["iter_id"],
            },
        },
        artifacts={
            "systems": loop.inputs.artifacts[
                "systems"
            ],  # starting systems for model deviation
            "teacher_model": loop.inputs.artifacts[
                "teacher_model"
            ],  # model for exploration
            "init_data": loop.inputs.artifacts["init_data"],
            "iter_data": loop.inputs.artifacts["iter_data"],
        },
        key="--".join(
            ["iter-%s" % stage_scheduler.outputs.parameters["iter_id"], "explore-block"]
        ),
        # when="%s == false" % (stage_scheduler.outputs.parameters["converged"]),
    )
    loop.add(expl_dist_blk)

    # next iteration
    next_parameters = {
        "converged": stage_scheduler.outputs.parameters["converged"],
        "block_id": stage_scheduler.outputs.parameters["next_iter_id"],
        "type_map": loop.inputs.parameters["type_map"],
        "mass_map": loop.inputs.parameters["mass_map"],
        "converge_config": loop.inputs.parameters["converge_config"],
        "conf_filters_conv": loop.inputs.parameters["conf_filters_conv"],
        "numb_models": loop.inputs.parameters["numb_models"],
        "template_script": loop.inputs.parameters["template_script"],
        "train_config": loop.inputs.parameters["train_config"],
        "explore_config": loop.inputs.parameters["explore_config"],
        "inference_config": loop.inputs.parameters["inference_config"],
        "type_map_train": loop.inputs.parameters["type_map_train"],
        "scheduler": stage_scheduler.outputs.parameters["scheduler"],
        "converged": expl_dist_blk.outputs.parameters["converged"],
    }
    next_step = Step(
        name=name + "-exploration-finetune-next",
        template=loop,
        parameters=next_parameters,
        artifacts={
            "systems": loop.inputs.artifacts["systems"],
            "teacher_model": loop.inputs.artifacts["teacher_model"],
            "iter_data": expl_dist_blk.outputs.artifacts["iter_data"],
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
    loop.outputs.artifacts["dist_model"].from_expression = if_expression(
        _if=stage_scheduler.outputs.parameters["converged"],
        _then=expl_dist_blk.outputs.artifacts["dist_model"],
        _else=next_step.outputs.artifacts["dist_model"],
    )

    loop.outputs.artifacts["iter_data"].from_expression = if_expression(
        _if=stage_scheduler.outputs.parameters["converged"],
        _then=expl_dist_blk.outputs.artifacts["iter_data"],
        _else=next_step.outputs.artifacts["iter_data"],
    )

    return loop
