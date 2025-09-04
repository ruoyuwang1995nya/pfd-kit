import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
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

from pfd.constants import (
    lmp_index_pattern,
)
from pfd.utils.step_config import (
    init_executor,
)
from pfd.utils.step_config import normalize as normalize_step_dict


class PrepRunExpl(Steps):
    def __init__(
        self,
        name: str,
        prep_op: Type[OP],
        run_op: Type[OP],
        prep_config: Optional[dict] = None,
        run_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        run_config = normalize_step_dict({}) if run_config is None else run_config
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "explore_config": InputParameter(),
            "expl_task_grp": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "trajs": OutputArtifact(),
            #"model_devis": OutputArtifact(),
            #"plm_output": OutputArtifact(),
            "optional_outputs": OutputArtifact(),
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

        self._keys = ["prep-expl", "run-expl"]
        self.step_keys = {}
        ii = "prep-expl"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-expl"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_expl(
            self,
            self.step_keys,
            prep_op,
            run_op,
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


def _prep_run_expl(
    prep_run_steps,
    step_keys,
    prep_op: Type[OP],
    run_op: Type[OP],
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

    print(prep_run_steps.inputs.parameters)
    prep_lmp = Step(
        "prep-expl",
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "expl_task_grp": prep_run_steps.inputs.parameters["expl_task_grp"],
        },
        artifacts={},
        key=step_keys["prep-expl"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_lmp)

    run_lmp = Step(
        "run-expl",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path"],
                output_artifact=[
                    "log",
                    "traj",
                    "optional_output",
                ],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_lmp.outputs.parameters["task_names"],
            "config": prep_run_steps.inputs.parameters["explore_config"],
        },
        artifacts={
            "task_path": prep_lmp.outputs.artifacts["task_paths"],
            "models": prep_run_steps.inputs.artifacts["models"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_lmp.outputs.parameters["task_names"]),
            format=lmp_index_pattern,
        ),
        key=step_keys["run-expl"],
        executor=run_executor,
        **run_config,
    )
    prep_run_steps.add(run_lmp)

    prep_run_steps.outputs.parameters[
        "task_names"
    ].value_from_parameter = prep_lmp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_lmp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["trajs"]._from = run_lmp.outputs.artifacts["traj"]
    prep_run_steps.outputs.artifacts[
        "optional_outputs"
    ]._from = run_lmp.outputs.artifacts["optional_output"]

    return prep_run_steps