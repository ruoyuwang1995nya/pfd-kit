import dpdata
from pathlib import Path
import random
from pathlib import (
    Path,
)
from typing import List, Dict, Tuple, Union, Optional
import random
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("collect.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CollectData(OP):
    """Collect data for further operations, return dpdata.Multisystems

    Args:
        OP (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "additional_systems": Artifact(List[Path], optional=True),
                "additional_multi_systems": Artifact(List[Path], optional=True),
                "type_map": Parameter(Union[List[str], None], default=None),
                "optional_parameters": Parameter(Dict, default={}),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "multi_systems": Artifact(List[Path]),
                "test_systems": Artifact(List[Path], optional=True),
            }
        )

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
            - `lmp_task_grp` : (`BigParameter(Path)`) Can be pickle loaded as a ExplorationTaskGroup. Definitions for LAMMPS tasks

        Returns
        -------
        op : dict
            Output dict with components:
            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the LAMMPS simulation. The order fo the Paths should be consistent with `op["task_names"]`
        """
        # List of systems
        systems = ip["systems"]
        if additional_systems := ip.get("additional_systems"):
            systems.extend(additional_systems)
        # List of multi_systems
        if multi_system := ip.get("additional_multi_systems"):
            print("Multi_sys is", multi_system)
        else:
            multi_system = []
        # Collects various types of data
        type_map = ip["type_map"]
        optional_parameters = ip["optional_parameters"]
        labeled = optional_parameters.get("labeled_data", False)
        test_size = optional_parameters.get("test_size")
        multi_sys_name = optional_parameters.pop("multi_sys_name", "multi_system")
        multi_sys = dpdata.MultiSystems()

        def read_sys(
            sys_path: Union[Path, str],
            labeled: bool = False,
            fmt: Optional[str] = None,
            type_map: Optional[List[str]] = None,
        ):
            if fmt is None:
                fmt = get_sys_fmt(sys_path)
            if not labeled:
                sys = dpdata.System(sys_path, fmt, type_map=type_map)
            else:
                sys = dpdata.LabeledSystem(sys_path, fmt, type_map=type_map)
            return sys

        if sample_param := optional_parameters.get("sample_conf"):
            for sys_path in [systems[ii] for ii in sample_param["confs"]]:
                logging.info("-----------------------------------")
                logging.info("## Collecting system: %03s" % sys_path)
                # logging.info("-----------------")
                sys = sample_sys(
                    read_sys(sys_path, labeled=labeled, type_map=type_map),
                    sample_param["n_sample"],
                )
                logging.info("%d frames collected" % sys.get_nframes())
                multi_sys.append(sys)
        else:
            for sys_path in systems:
                logging.info("-----------------------------------")
                logging.info("## Collecting system: %03s\n" % sys_path)
                # logging.info("-----------------")
                multi_sys.append(read_sys(sys_path, labeled=labeled, type_map=type_map))

        if test_size:
            multi_sys, test_sys, _ = multi_sys.train_test_split(
                test_size=test_size, seed=random.randint(1, 100000)
            )
        else:
            test_sys = dpdata.MultiSystems()

        multi_sys.to("deepmd/npy", multi_sys_name)
        test_multi_sys_name = "%s_test" % multi_sys_name
        test_sys.to("deepmd/npy", test_multi_sys_name)
        multi_system.append(Path(multi_sys_name))
        logging.info("-----------------------------------")
        logging.info("Save to dpdata.MultiSystems: %s" % multi_sys_name)
        logging.info("%d frames collected" % multi_sys.get_nframes())
        # ensure multi_sys_name exist
        Path(multi_sys_name).mkdir(exist_ok=True)
        return OPIO(
            {
                "multi_systems": multi_system,
                "systems": sorted(
                    [path for path in Path(multi_sys_name).iterdir() if path.is_dir()],
                    key=lambda p: p.name,
                ),
                "test_systems": sorted(
                    [
                        path
                        for path in Path(test_multi_sys_name).iterdir()
                        if path.is_dir()
                    ],
                    key=lambda p: p.name,
                )
                if Path(test_multi_sys_name).is_dir()
                else [],
            }
        )


def get_sys_fmt(sys_path: Union[Path, str]):
    "return dpdata format"
    if isinstance(sys_path, str):
        sys_path = Path(sys_path)
    if (sys_path / "type.raw").is_file():
        return "deepmd/npy"

    elif sys_path.is_file() and sys_path.suffix == ".dump":
        return "lammps/dump"
    else:
        raise NotImplementedError("Unknown data format")


def sample_sys(sys: dpdata.System, n_sample: int = 1):
    n_frame = sys.get_nframes()
    if n_sample >= n_frame:
        return sys
    else:
        frame_ls = [ii for ii in range(n_frame)]
        random.shuffle(frame_ls)
        return sys.sub_system(frame_ls[:n_sample])
