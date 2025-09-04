import dis
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
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)

from pfd.op import ModelTestOP
from pfd.utils.step_config import init_executor


class DataGen(Steps):
    """
    A class to represent the DataGen operation.

    Attributes:
        name (str): The name of the DataGen operation.
        pert_gen_op (Type[OP]): The perturbation generation operation.
        prep_run_fp_op (OPTemplate): The preparation and run force field operation template.
        collect_data_op (Type[OP]): The data collection operation.
        pert_gen_step_config (dict): Configuration for the perturbation generation step.
        collect_data_config (dict): Configuration for the data collection step.
        upload_python_packages (Optional[List[os.PathLike]]): List of Python packages to upload.
    """

    def __init__(
        self,
        name: str,
        pert_gen_op: Type[OP],
        prep_run_fp_op: OPTemplate,
        collect_data_op: Type[OP],
        pert_gen_step_config: dict,
        collect_data_config: dict,
        upload_python_packages: Optional[List[os.PathLike]] = None
    ):
        """
        Initializes the DataGen class.

        Args:
            name (str): The name of the DataGen operation.
            pert_gen_op (Type[OP]): The perturbation generation operation.
            prep_run_fp_op (OPTemplate): The preparation and run force field operation template.
            collect_data_op (Type[OP]): The data collection operation.
            pert_gen_step_config (dict): Configuration for the perturbation generation step.
            collect_data_config (dict): Configuration for the data collection step.
            upload_python_packages (Optional[List[os.PathLike]]): List of Python packages to upload.
        """
        
        self._input_parameters = {
            "type_map": InputParameter(type=Optional[List],value=None),
            "fp_config": InputParameter(),
            "pert_config": InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "multi_systems": OutputArtifact()
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
        self = _data_gen(
            self,
            name,
            pert_gen_op,
            prep_run_fp_op,
            collect_data_op,
            pert_gen_step_config,
            collect_data_config,
            upload_python_packages
        )

    @property
    def input_parameters(self):
        """
        Returns the input parameters.

        Returns:
            dict: The input parameters.
        """
        return self._input_parameters

    @property
    def input_artifacts(self):
        """
        Returns the input artifacts.

        Returns:
            dict: The input artifacts.
        """
        return self._input_artifacts

    @property
    def output_parameters(self):
        """
        Returns the output parameters.

        Returns:
            dict: The output parameters.
        """
        return self._output_parameters

    @property
    def output_artifacts(self):
        """
        Returns the output artifacts.

        Returns:
            dict: The output artifacts.
        """
        return self._output_artifacts
    
def _data_gen(
    steps,
    name: str,
    pert_gen_op: Type[OP],
    prep_run_fp_op: OPTemplate,
    collect_data_op: Type[OP],
    pert_gen_step_config: dict,
    collect_data_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
):
    """
    Creates the steps for the DataGen operation.

    Args:
        steps: The steps object to add the steps to.
        name (str): The name of the DataGen operation.
        pert_gen_op (Type[OP]): The perturbation generation operation.
        prep_run_fp_op (OPTemplate): The preparation and run force field operation template.
        collect_data_op (Type[OP]): The data collection operation.
        pert_gen_step_config (dict): Configuration for the perturbation generation step.
        collect_data_config (dict): Configuration for the data collection step.
        upload_python_packages (Optional[List[os.PathLike]]): List of Python packages to upload.

    Returns:
        The steps object with the added steps.
    """
    pert_gen_step_config = deepcopy(pert_gen_step_config)
    pert_gen_template_config = pert_gen_step_config.pop("template_config")
    pert_gen_executor = init_executor(pert_gen_step_config.pop("executor"))
    
    collect_data_config = deepcopy(collect_data_config)
    collect_data_template_config = collect_data_config.pop("template_config")
    collect_data_executor = init_executor(collect_data_config.pop("executor"))
    
    pert_gen = Step(
        name + "-pert-gen",
        template=PythonOPTemplate(
            pert_gen_op,
            python_packages=upload_python_packages,
            **pert_gen_template_config
        ),
        parameters={"config": steps.inputs.parameters["pert_config"]},
        artifacts={"init_confs": steps.inputs.artifacts["init_confs"]},
        key="--".join(["init", "pert-gen"]),
        executor=pert_gen_executor,
        **pert_gen_step_config
    )
    steps.add(pert_gen)
    
    
    prep_run_fp = Step(
        name=name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": "init",
            "fp_config": steps.inputs.parameters["fp_config"],
            "type_map": steps.inputs.parameters["type_map"],
                },
        artifacts={
            "confs": pert_gen.outputs.artifacts["confs"]
                },
                key="--".join(["init", "prep-run-fp"]),
            )
    steps.add(prep_run_fp)
    
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
                "multi_sys_name": "datas",
                },
            "type_map": steps.inputs.parameters["type_map"],
                },
        artifacts={"systems": prep_run_fp.outputs.artifacts["labeled_data"]},
        key="--".join(["init", "collect-data"]),
        executor=collect_data_executor,
                **collect_data_config
            )
    steps.add(collect_data)
    steps.outputs.artifacts["multi_systems"]._from = collect_data.outputs.artifacts[
        "multi_systems"
    ]
    return steps