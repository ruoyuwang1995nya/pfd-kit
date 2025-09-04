import json
import os
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Dict,
    List,
    Set,
    Tuple,
)

import dargs
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    OPIOSign,
    TransientError,
)

from pfd.utils.chdir import (
    set_directory,
)


class RunFp(OP, ABC):
    r"""Execute a first-principles (FP) task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The FP
    command is exectuted from directory `task_name`. The
    `op["labeled_data"]` in `"deepmd/npy"` format (HF5 in the future)
    provided by `dpdata` will be created.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "model": Artifact(Path, optional=True),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "labeled_data": Artifact(Path),
                "extra_outputs": Artifact(List[Path]),
            }
        )

    @abstractmethod
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a FP task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        pass

    @abstractmethod
    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a FP task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        pass

    @abstractmethod
    def run_task(
        self,
        **kwargs,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        **kwargs
            Keyword args defined by the developer.
            The fp/run_config session of the input file will be passed to this function.

        Returns
        -------
        out_name: str
            The file name of the output data. Should be in dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """
        pass

    @staticmethod
    @abstractmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """
        pass

    @classmethod
    def normalize_config(cls, data: Dict = {}, strict: bool = True) -> Dict:
        r"""Normalized the argument.

        Parameters
        ----------
        data : Dict
            The input dict of arguments.
        strict : bool
            Strictly check the arguments.

        Returns
        -------
        data: Dict
            The normalized arguments.

        """
        ta = cls.args()
        base = dargs.Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `config`: (`dict`) The config of FP task. Should have `config['run']`, which defines the runtime configuration of the FP task.
            - `task_name`: (`str`) The name of task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepFp`.

        Returns
        -------
        Output dict with components:
        - `log`: (`Artifact(Path)`) The log file of FP.
        - `labeled_data`: (`Artifact(Path)`) The path to the labeled data in `"deepmd/npy"` format provided by `dpdata`.

        Raises
        ------
        TransientError
            On the failure of FP execution.
        FatalError
            When mandatory files are not found.
        """
        config = ip["config"]["run"] if ip["config"]["run"] is not None else {}
        config = type(self).normalize_config(config, strict=False)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        model = ip.get("model")
        input_files = self.input_files()
        input_files = [(Path(task_path) / ii).resolve() for ii in input_files]
        opt_input_files = self.optional_input_files()
        opt_input_files = [(Path(task_path) / ii).resolve() for ii in opt_input_files]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                if os.path.isfile(ii) or os.path.isdir(ii):
                    iname = ii.name
                    Path(iname).symlink_to(ii)
                else:
                    raise FatalError(f"cannot find file {ii}")

            for ii in opt_input_files:
                if os.path.isfile(ii) or os.path.isdir(ii):
                    iname = ii.name
                    Path(iname).symlink_to(ii)
            out_name, log_name = self.run_task(
                **config,
                model = model,
                )

        extra_outputs = []
        for fname in ip["config"]["extra_output_files"]:
            extra_outputs += list(work_dir.glob(fname))

        return OPIO(
            {
                "log": work_dir / log_name,
                "labeled_data": work_dir / out_name,
                "extra_outputs": extra_outputs,
            }
        )