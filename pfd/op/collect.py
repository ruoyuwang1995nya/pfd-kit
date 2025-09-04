from ase.io import read, write
from ase.io.formats import UnknownFileTypeError
from pathlib import Path
from typing import List, Dict
from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter
from pfd.utils.ase2xyz import train_test_split

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("collect.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CollectData(OP):
    """Collect and process molecular systems data for machine learning workflows.
    
    This operation aggregates multiple atomic systems, applies optional sampling,
    and converts them to dpdata.MultiSystems format for downstream ML training.
    Supports both labeled and unlabeled data with optional train/test splitting.
    
    Examples
    --------
    >>> collector = CollectData()
    >>> result = collector.execute({
    ...     "systems": [Path("system1"), Path("system2")],
    ...     "type_map": ["H", "O"],
    ...     "optional_parameters": {"test_size": 0.2}
    ... })
    """

    @classmethod
    def get_input_sign(cls):
        r"""Get the input signature for the operation.
        Returns:
        -----------
            OPIOSign: The input signature.
        """
        return OPIOSign(
            {
                "structures": Artifact(List[Path]), # systems in the form of extxyz files
                "pre_structures": Artifact(List[Path], optional=True), # all the previous systems
                "optional_structures": Artifact(List[Path], optional=True), # additional systems
                "optional_parameters": Parameter(Dict, default={}),
                "iter_id":Parameter(str,default=None)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "structures": Artifact(Path), 
                "test_structures": Artifact(Path, optional=True),
                "iter_structures": Artifact(Path),
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
            - `structures` : (`Artifact(List[Path])`) configurations collected in this iteration
            - `pre_structures` : (`Artifact(Path)`, optional) A single extxyz file, configurations collected in previous iterations at the CURRENT stage

        Returns
        -------
        op : dict
            Output dict with components:
            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the LAMMPS simulation. The order fo the Paths should be consistent with `op["task_names"]`
        """
        # structures collected in this iteration
        iter_structures = ip["structures"]
        pre_structures = ip.get("pre_structures")
        optional_structures = ip.get("optional_structures")
        # Collects various types of data
        optional_parameters = ip["optional_parameters"]
        test_size = optional_parameters.get("test_size",0.1)
        structures_name = optional_parameters.pop("structures", "structures")
        iter_id = ip.get("iter_id")

        structures = []
        for path in iter_structures:
            try:
                structures.extend(read(path,index=":"))
            except UnknownFileTypeError as e:
                logging.warning(f"Unknown file type for {path}: {e}")
                continue
        # label iteration id
        if iter_id:
            for atoms in structures:
                atoms.info['iter']=iter_id
        train_structures = structures
        if optional_structures:
            test_structures = optional_structures
        else:
            train_structures, test_structures = train_test_split(structures, test_size=test_size,random_state=1)
        
        if pre_structures:
            pre_structures_ls=[]
            for path in pre_structures:
                pre_structures_ls.extend(read(path,index=":"))
                train_structures.extend(pre_structures_ls)
            # append the current iteration to iter data
            pre_structures_ls.extend(structures)
        else:
            pre_structures_ls = structures

        # structures for training
        train_structures_name = "%s_train.extxyz" % structures_name
        write(train_structures_name, train_structures)

        all_structures_name = "%s_all.extxyz" % structures_name
        write(all_structures_name, pre_structures_ls)

        # structures for testing
        test_structures_name = "%s_test.extxyz" % structures_name
        write(test_structures_name, test_structures)
        logging.info("-----------------------------------")
        logging.info("Save to extxyz file: %s" % structures_name)
        logging.info("%d frames collected" % len(structures))
        return OPIO(
            {
                "structures": train_structures_name,
                "iter_structures": all_structures_name,
                "test_structures": test_structures_name
            }
        )
