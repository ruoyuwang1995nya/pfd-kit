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


from dpgen2.utils.step_config import init_executor


class FineTune(Steps):
    def __init__(
        self,
        name: str,
        pert_gen_op: Type[OP],
        prep_run_fp_op: OPTemplate,
        collect_data_op: Type[OP],
        prep_run_dp_train_op: OPTemplate,
        expl_finetune_loop_op: OPTemplate,
        pert_gen_step_config: dict,
        collect_data_step_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
        skip_exploration=False,
        skip_aimd: bool = True,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "mass_map": InputParameter(),
            "pert_config": InputParameter(),  # Total input parameter file: to be changed in the future
            # md exploration
            "numb_models": InputParameter(type=int),
            "explore_config": InputParameter(),
            "conf_selector": InputParameter(),
            "conf_filters_conv": InputParameter(),
            "scheduler": InputParameter(),
            # training
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            # test
            "inference_config": InputParameter(),
            # fp calculation for labeling
            "fp_config": InputParameter(),
            # fp exploration
            "aimd_config": InputParameter(),
            "aimd_sample_conf": InputParameter(type=Optional[Dict], value={}),
            "collect_data_config": InputParameter(),
            "converge_config": InputParameter(),
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "expl_models": InputArtifact(optional=True),
            "init_models": InputArtifact(),
            "iter_data": InputArtifact(optional=True),
            "init_data": InputArtifact(optional=True),
            "validation_data": InputArtifact(optional=True),
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "fine_tuned_model": OutputArtifact(),
            "fine_tune_report": OutputArtifact(),
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
        self = _fine_tune_cl(
            self,
            name,
            pert_gen_op=pert_gen_op,
            prep_run_fp_op=prep_run_fp_op,
            prep_run_dp_train_op=prep_run_dp_train_op,
            collect_data_op=collect_data_op,
            expl_finetune_loop_op=expl_finetune_loop_op,
            pert_gen_step_config=pert_gen_step_config,
            collect_data_config=collect_data_step_config,
            upload_python_packages=upload_python_packages,
            skip_exploration=skip_exploration,
            skip_aimd=skip_aimd,
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
    name: str,
    pert_gen_op: Type[OP],
    prep_run_fp_op: OPTemplate,
    prep_run_dp_train_op: OPTemplate,
    expl_finetune_loop_op: OPTemplate,
    collect_data_op: Type[OP],
    pert_gen_step_config: dict,
    collect_data_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    skip_exploration: bool = True,
    skip_aimd: bool = True,
):
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))

    collect_data_step_config = deepcopy(collect_data_config)
    collect_data_template_config = collect_data_step_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_step_config.pop("executor"))

    pert_gen = Step(
        name + "-pert-gen",
        template=PythonOPTemplate(
            pert_gen_op,
            python_packages=upload_python_packages,
            **pert_gen_template_config
        ),
        parameters={"config": ft_steps.inputs.parameters["pert_config"]},
        artifacts={"init_confs": ft_steps.inputs.artifacts["init_confs"]},
        key="--".join(["init", "pert-gen"]),
        executor=pert_gen_executor,
        **pert_gen_step_config
    )
    ft_steps.add(pert_gen)

    if skip_exploration is True:
        # if skip AIMD exploration
        if skip_aimd is True:
            loop_iter_data = ft_steps.inputs.artifacts.get("iter_data")
        # if execute AIMD exploration
        else:
            sample_conf_aimd = Step(
                name=name + "-sample-aimd",
                template=PythonOPTemplate(
                    collect_data_op,
                    python_packages=upload_python_packages,
                    **collect_data_template_config
                ),
                parameters={
                    "optional_parameters": ft_steps.inputs.parameters[
                        "aimd_sample_conf"
                    ],
                    "type_map": ft_steps.inputs.parameters["type_map"],
                },
                artifacts={"systems": pert_gen.outputs.artifacts["pert_sys"]},
                key="--".join(["init", "sample-aimd"]),
                executor=collect_data_executor,
                **collect_data_step_config
            )
            ft_steps.add(sample_conf_aimd)

            prep_run_fp = Step(
                name=name + "-prep-run-fp",
                template=prep_run_fp_op,
                parameters={
                    "block_id": "init",
                    "fp_config": ft_steps.inputs.parameters["aimd_config"],
                    "type_map": ft_steps.inputs.parameters["type_map"],
                },
                artifacts={
                    "confs": sample_conf_aimd.outputs.artifacts["multi_systems"]
                },
                key="--".join(["init", "prep-run-fp"]),
            )
            ft_steps.add(prep_run_fp)
            collect_data = Step(
                name=name + "-collect-aimd",
                template=PythonOPTemplate(
                    collect_data_op,
                    python_packages=upload_python_packages,
                    **collect_data_template_config
                ),
                parameters={
                    "optional_parameters": {
                        "labeled_data": True,
                        "multi_sys_name": "init",
                    },
                    "type_map": ft_steps.inputs.parameters["type_map"],
                },
                artifacts={"systems": prep_run_fp.outputs.artifacts["labeled_data"]},
                key="--".join(["init", "collect-data"]),
                executor=collect_data_executor,
                **collect_data_step_config
            )
            ft_steps.add(collect_data)
            loop_iter_data = collect_data.outputs.artifacts["multi_systems"]

        # init model training
        prep_run_ft = Step(
            name + "-prep-run-dp-train",
            template=prep_run_dp_train_op,
            parameters={
                "block_id": "init",
                "train_config": ft_steps.inputs.parameters["train_config"],
                "numb_models": ft_steps.inputs.parameters["numb_models"],
                "template_script": ft_steps.inputs.parameters["template_script"],
                "run_optional_parameter": {
                    "mixed_type": False,
                    "finetune_mode": "finetune",
                },
            },
            artifacts={
                "init_models": ft_steps.inputs.artifacts["init_models"],
                "init_data": ft_steps.inputs.artifacts["init_data"],
                "iter_data": loop_iter_data,  # collect_data.outputs.artifacts["multi_systems"]#steps.inputs.artifacts["iter_data"],
            },
            key="--".join(["init", "prep-run-train"]),
        )
        ft_steps.add(prep_run_ft)
        expl_models = prep_run_ft.outputs.artifacts.get("models")
    # if skip initial model training
    else:
        loop_iter_data = ft_steps.inputs.artifacts.get("iter_data")
        expl_models = ft_steps.inputs.artifacts.get("expl_models")

    loop = Step(
        name="ft-loop",
        template=expl_finetune_loop_op,
        parameters={
            "fp_config": ft_steps.inputs.parameters["fp_config"],
            "type_map": ft_steps.inputs.parameters["type_map"],
            "mass_map": ft_steps.inputs.parameters["mass_map"],
            "conf_selector": ft_steps.inputs.parameters["conf_selector"],
            "conf_filters_conv": ft_steps.inputs.parameters["conf_filters_conv"],
            "numb_models": ft_steps.inputs.parameters["numb_models"],
            "template_script": ft_steps.inputs.parameters["template_script"],
            "train_config": ft_steps.inputs.parameters["train_config"],
            "inference_config": ft_steps.inputs.parameters["inference_config"],
            "explore_config": ft_steps.inputs.parameters["explore_config"],
            "dp_test_validation_config": {},
            "converge_config": ft_steps.inputs.parameters["converge_config"],
            "scheduler": ft_steps.inputs.parameters["scheduler"],
        },
        artifacts={
            "systems": pert_gen.outputs.artifacts[
                "pert_sys"
            ],  # starting systems for model deviation
            "current_model": expl_models,  # ft_steps.inputs.artifacts["expl_models"],
            "init_model": ft_steps.inputs.artifacts[
                "init_models"
            ],  # starting point for finetune
            "init_data": ft_steps.inputs.artifacts[
                "init_data"
            ],  # initial data for model finetune
            "iter_data": loop_iter_data,  # ft_steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % "test", "-fp"]),
    )
    ft_steps.add(loop)
    ft_steps.outputs.artifacts["fine_tuned_model"]._from = loop.outputs.artifacts[
        "ft_model"
    ][0]
    ft_steps.outputs.artifacts["fine_tune_report"]._from = loop.outputs.artifacts[
        "iter_data"
    ]
