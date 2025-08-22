import os
from copy import (
    deepcopy,
)
from turtle import st
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
from typeguard import value
from pfd import train
from pfd.utils.step_config import init_executor
from pfd.op import EvalConv, ModelTestOP, StageScheduler


class ExplTrainBlock(Steps):
    def __init__(
        self,
        name: str,
        prep_run_explore_op: OPTemplate,
        prep_run_fp_op: OPTemplate,
        collect_data_op: Type[OP],
        select_confs_op: Type[OP],
        prep_run_train_op: Type[OP],
        test_op: Type[OP],
        collect_data_config: dict,
        test_config: dict,
        train_config: dict,
        select_confs_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "expl_tasks": InputParameter(),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "fp_config": InputParameter(),
            "collect_data_config": InputParameter(),
            "evaluate_config": InputParameter(value={}),
            "select_confs_config": InputParameter(value={}),
            "conf_selector": InputParameter(value=None),
            #"conf_filters_conv": InputParameter(value=None),
        }
        self._input_artifacts = {
            #"systems": InputArtifact(),  # starting systems for model deviation
            "expl_model": InputArtifact(optional=True),  # model for exploration
            "init_model": InputArtifact(optional=True),  # starting point for finetune
            "init_data": InputArtifact(optional=True),  # initial data for model finetune, would be a list of extxyz files
            "iter_data": InputArtifact(
                optional=True
            ),  # datas collected during previous exploration
        }
        self._output_parameters = {
            "converged": OutputParameter(),
            "report": OutputParameter(default=None),
        }
        self._output_artifacts = {
            "model": OutputArtifact(),
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
        self = _expl_tr_blk(
            self,
            name=name,
            prep_run_explore_op=prep_run_explore_op,
            prep_run_fp_op=prep_run_fp_op,
            prep_run_train_op=prep_run_train_op,
            select_confs_op=select_confs_op,
            test_op=test_op,
            train_config=train_config,
            collect_data_op=collect_data_op,
            collect_data_config=collect_data_config,
            test_config=test_config,
            select_confs_config=select_confs_config,
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

def _expl_tr_blk(
    steps,
    name: str,
    prep_run_explore_op: OPTemplate,
    prep_run_fp_op: OPTemplate,
    prep_run_train_op: Type[OP],
    test_op: Type[OP],
    select_confs_op: Type[OP],
    collect_data_op: Type[OP],
    test_config: dict,
    train_config: dict,
    select_confs_config: dict,
    collect_data_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):

    test_config = deepcopy(test_config)
    test_template_config = test_config.pop("template_config")
    test_executor = init_executor(test_config.pop("executor"))

    train_config = deepcopy(train_config)
    train_template_config = train_config.pop("template_config")
    train_executor = init_executor(train_config.pop("executor"))

    # essentially for utillity
    collect_data_config = deepcopy(collect_data_config)
    collect_data_template_config = collect_data_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_config.pop("executor"))
    
    # essentially for utillity
    select_confs_config = deepcopy(select_confs_config)
    select_confs_template_config = select_confs_config.pop("template_config")
    select_confs_executor = init_executor(select_confs_config.pop("executor"))
    

    prep_run_explore = Step(
        name + "-prep-run-explore",
        template=prep_run_explore_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "explore_config": steps.inputs.parameters["explore_config"],
            "expl_task_grp": steps.inputs.parameters["expl_tasks"],
        },
        artifacts={"models": steps.inputs.artifacts["expl_model"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-explore"]),
    )
    steps.add(prep_run_explore)

    select_confs = Step(
        name + "-select-confs",
        template=PythonOPTemplate(
            select_confs_op,
            #output_artifact_archive={"confs": None},
            python_packages=upload_python_packages,
            **select_confs_template_config,
        ),
        parameters={
            "conf_selector": steps.inputs.parameters["conf_selector"],
            "optional_parameters": steps.inputs.parameters["select_confs_config"],
        },
        artifacts={"confs": prep_run_explore.outputs.artifacts["trajs"]},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "select-confs"]),
        executor=select_confs_executor,
        **select_confs_config,
    )
    steps.add(select_confs)

    prep_run_fp = Step(
        name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "fp_config": steps.inputs.parameters["fp_config"],
        },
        artifacts={
            "confs": select_confs.outputs.artifacts["confs"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "prep-run-fp"]),
    )
    steps.add(prep_run_fp)

    collect_data_train = Step(
        name + "-collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "optional_parameters": steps.inputs.parameters["collect_data_config"],
        },
        artifacts={
            "structures": prep_run_fp.outputs.artifacts["labeled_data"],
            "pre_structures": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "collect-data"]),
        executor=collect_data_executor,
        **collect_data_config,
    )
    steps.add(collect_data_train)

    prep_run_train = Step(
        name + "-prep-run-train",
        template=PythonOPTemplate(
            prep_run_train_op,
            python_packages=upload_python_packages,
            **train_template_config,
        ),
        parameters={
            "train_config": steps.inputs.parameters["train_config"],
            "template_script": steps.inputs.parameters["template_script"],
        },
        artifacts={
            "init_data": steps.inputs.artifacts["init_data"],
            "init_model": steps.inputs.artifacts["init_model"],
            "iter_data": collect_data_train.outputs.artifacts["structures"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "train"]),
        executor=train_executor,
        **train_config,
    )
    steps.add(prep_run_train)

    # the exploration steps for validation
    evaluate = Step(
        name + "-test-model",
        template=PythonOPTemplate(
            test_op,
            python_packages=upload_python_packages,
            **test_template_config,
        ),
        parameters={
            "config": steps.inputs.parameters["evaluate_config"]
        },
        artifacts={
            "structures": collect_data_train.outputs.artifacts["test_structures"],
            "model": prep_run_train.outputs.artifacts["model"],
        },
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "evaluate"]),
        executor=test_executor,
    )
    steps.add(evaluate)

    steps.outputs.artifacts["iter_data"]._from = collect_data_train.outputs.artifacts[
        "iter_structures"
    ]
    steps.outputs.artifacts["model"]._from = prep_run_train.outputs.artifacts[
        "model"
    ]
    steps.outputs.artifacts["test_report"]._from = evaluate.outputs.artifacts[
        "test_report"
    ]
    steps.outputs.parameters[
        "converged"
    ].value_from_parameter = evaluate.outputs.parameters["converged"]
    steps.outputs.parameters[
        "report"
    ].value_from_parameter = evaluate.outputs.parameters["report"]
    return steps

class ExplTrainLoop(Steps):
    def __init__(
        self,
        name: str,
        stage_scheduler_op: Type[OP],
        expl_ft_blk_op: OPTemplate,
        scheduler_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(value="000"),
            "conf_selector": InputParameter(),
            "fp_config": InputParameter(),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "evaluate_config": InputParameter(value={}),
            "collect_data_config": InputParameter(value={}),
            #"schedule_config": InputParameter(value={}),
            "scheduler": InputParameter(),
            "converged": InputParameter(value=False),
            "report": InputParameter(value=None),
        }
        self._input_artifacts = {
            #"systems": InputArtifact(),  # starting systems for model deviation
            "current_model": InputArtifact(optional=True),  # current model
            "expl_model": InputArtifact(optional=True),  # model for exploration
            "init_model": InputArtifact(optional=True),  # starting point for finetune
            "init_data": InputArtifact(optional=True),  # initial data for model fine-tuning, data collected from PREVIOUS stages as well as initial data
            "iter_data": InputArtifact(
                optional=True
            ),  # data collected during previous iterations at the CURRENT exploration stage
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
            stage_scheduler_op=stage_scheduler_op,
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


def _loop(
    loop,  # the loop Steps
    name: str,
    stage_scheduler_op: Type[OP],
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
            stage_scheduler_op,
            python_packages=upload_python_packages,
            **scheduler_template_config,
        ),
        parameters={
            "converged": loop.inputs.parameters["converged"],
            "scheduler": loop.inputs.parameters["scheduler"],
            "report": loop.inputs.parameters["report"],
        },
        artifacts={
            "init_model": loop.inputs.artifacts["init_model"],
            "current_model": loop.inputs.artifacts["current_model"],
            "expl_model": loop.inputs.artifacts["expl_model"],
            "iter_data": loop.inputs.artifacts["iter_data"],
        },
        key="--".join(["iter-%s" % loop.inputs.parameters["block_id"], "scheduler"]),
        executor=scheduler_executor,
        **scheduler_config,
    )
    loop.add(stage_scheduler)

    expl_ft_blk = Step(
        name=name + "-exploration-train",
        template=expl_ft_blk_op,
        parameters={
            "block_id": "iter-%s" % stage_scheduler.outputs.parameters["iter_id"],
            "expl_tasks": stage_scheduler.outputs.parameters["task_grp"],
            "conf_selector": loop.inputs.parameters["conf_selector"],
            "template_script": loop.inputs.parameters["template_script"],
            "train_config": stage_scheduler.outputs.parameters["train_config"],
            "explore_config": loop.inputs.parameters["explore_config"],
            "fp_config": loop.inputs.parameters["fp_config"],
            "evaluate_config": loop.inputs.parameters["evaluate_config"],
            "collect_data_config": loop.inputs.parameters["collect_data_config"],
        },
        artifacts={
            "expl_model": stage_scheduler.outputs.artifacts[
                "expl_model"
            ],  # model for exploration
            "init_model": stage_scheduler.outputs.artifacts[
                "init_model"
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
        "conf_selector": loop.inputs.parameters["conf_selector"],
        "template_script": loop.inputs.parameters["template_script"],
        "train_config": loop.inputs.parameters["train_config"],
        "explore_config": loop.inputs.parameters["explore_config"],
        "fp_config": loop.inputs.parameters["fp_config"],
        "collect_data_config": loop.inputs.parameters["collect_data_config"],
        "evaluate_config": loop.inputs.parameters["evaluate_config"],
        "scheduler": stage_scheduler.outputs.parameters["scheduler"],
        "converged": expl_ft_blk.outputs.parameters["converged"],
        "report": expl_ft_blk.outputs.parameters["report"],
    }
    next_step = Step(
        name=name + "-exploration-train-next",
        template=loop,
        parameters=next_parameters,
        artifacts={
            "init_model": stage_scheduler.outputs.artifacts["init_model"],
            "current_model": expl_ft_blk.outputs.artifacts["model"],
            "expl_model": stage_scheduler.outputs.artifacts["expl_model"],
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
    loop.outputs.parameters["report"].value_from_parameter = stage_scheduler.outputs.parameters["report"]
    loop.outputs.artifacts["ft_model"]._from = stage_scheduler.outputs.artifacts["current_model"]
    loop.outputs.artifacts["iter_data"]._from = stage_scheduler.outputs.artifacts["iter_data"]
    return loop
