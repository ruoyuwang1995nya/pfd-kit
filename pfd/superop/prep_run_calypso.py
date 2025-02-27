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
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    calypso_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


## reuse the steps, but discard the model deviation steps
class PrepRunCaly(Steps):
    def __init__(
        self,
        name: str,
        prep_caly_input_op: Type[OP],
        caly_evo_step_op: Union[OPTemplate, OP],
        # prep_caly_model_devi_op: Type[OP],
        # run_caly_model_devi_op: Type[OP],
        expl_mode: str = "default",
        prep_config: Optional[dict] = None,
        run_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        run_config = normalize_step_dict({}) if run_config is None else run_config
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "expl_task_grp": InputParameter(),
            "explore_config": InputParameter(),
            "type_map": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            # "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "trajs": OutputArtifact(),
            "model_devis": OutputArtifact(optional=True),
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

        # TODO: RunModelDevi
        self._keys = [
            "prep-caly-input",
            "caly-evo-step-{{item}}",
            "prep-caly-model-devi",
            "run-caly-model-devi",
        ]
        self.step_keys = {}
        ii = "prep-caly-input"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "caly-evo-step-{{item}}"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "prep-caly-model-devi"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-caly-model-devi"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _prep_run_caly(
            self,
            self.step_keys,
            prep_caly_input_op,
            caly_evo_step_op,
            # prep_caly_model_devi_op,
            # run_caly_model_devi_op,
            expl_mode=expl_mode,
            prep_config=prep_config,
            run_config=run_config,
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


def _prep_run_caly(
    prep_run_caly_steps: Steps,
    step_keys: Dict[str, Any],
    prep_caly_input_op: Type[OP],
    caly_evo_step_op: Union[OPTemplate, OP],
    # prep_caly_model_devi_op: Type[OP],
    # run_caly_model_devi_op: Type[OP],
    expl_mode: str = "default",
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    # prep caly input files
    prep_caly_input = Step(
        "prep-caly-input",
        template=PythonOPTemplate(
            prep_caly_input_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "caly_task_grp": prep_run_caly_steps.inputs.parameters["expl_task_grp"],
        },
        artifacts={},
        key=step_keys["prep-caly-input"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_caly_steps.add(prep_caly_input)

    temp_value = None
    if expl_mode == "default":
        caly_evo_step_config = prep_config
        caly_evo_step_executor = None
        caly_evo_step_slice_config = {}
        template = caly_evo_step_op
    elif expl_mode == "merge":
        caly_evo_step_config = run_config
        caly_evo_step_executor = run_executor
        caly_evo_step_slice_config = deepcopy(template_slice_config)
        template = PythonOPTemplate(
            caly_evo_step_op,  # type: ignore
            python_packages=upload_python_packages,
            **run_template_config,
        )  # type: ignore
    else:
        raise KeyError(f"Unknown expl mode `{expl_mode}`")

    caly_evo_step = Step(
        "caly-evo-step",
        template=template,  # type: ignore
        slices=Slices(
            input_parameter=[
                "task_name",
            ],
            input_artifact=[
                "input_file",
                "results",
                "step",
                "opt_results_dir",
                "caly_run_opt_file",
                "caly_check_opt_file",
            ],
            output_artifact=["traj_results"],
            **caly_evo_step_slice_config,
        ),
        parameters={
            "block_id": prep_run_caly_steps.inputs.parameters["block_id"],
            "expl_config": prep_run_caly_steps.inputs.parameters["explore_config"],
            "task_name": prep_caly_input.outputs.parameters["task_names"],
            "iter_num": "{{item}}",
        },
        artifacts={
            "models": prep_run_caly_steps.inputs.artifacts["models"],
            "input_file": prep_caly_input.outputs.artifacts["input_dat_files"],
            "caly_run_opt_file": prep_caly_input.outputs.artifacts[
                "caly_run_opt_files"
            ],
            "caly_check_opt_file": prep_caly_input.outputs.artifacts[
                "caly_check_opt_files"
            ],
            "results": temp_value,
            "step": temp_value,
            "opt_results_dir": temp_value,
            "qhull_input": temp_value,
        },
        key=step_keys["caly-evo-step-{{item}}"],
        with_param=argo_range(prep_caly_input.outputs.parameters["ntasks"]),  # type: ignore
        executor=caly_evo_step_executor,
        **caly_evo_step_config,
    )
    prep_run_caly_steps.add(caly_evo_step)

    prep_run_caly_steps.outputs.artifacts[
        "trajs"
    ]._from = caly_evo_step.outputs.artifacts["traj_results"]

    return prep_run_caly_steps
