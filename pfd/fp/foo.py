import json
import logging
import os
import re
from pathlib import (
    Path,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from dargs import (
    Argument,
    dargs,
)

from pfd.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
    fp_task_pattern
)

from ase.io import write,read


from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)
from .vasp_input import (
    VaspInputs,
    make_kspacing_kpoints,
)
from pfd.utils import (
    set_directory)
from ase import Atoms
from ase.io import read

foo_conf_name = "structure.extxyz"
foo_log_name = "foo.log"
#foo_out_name = "structure.extxyz"


class PrepFoo(PrepFp):
    def _create_tasks(
        self,
        confs: List[Atoms],
        config: Dict,
        **kwargs,
    ):
        counter = 0
        task_names = []
        task_paths = []
        inputs = config["inputs"]
        for ii in range(len(confs)):
            ss = confs[ii]
            # loop over frames
            nn, pp = self._exec_one_frame(counter, inputs, ss)
            task_names.append(nn)
            task_paths.append(pp)
            counter += 1

        return task_names, task_paths

    def _exec_one_frame(
        self,
        idx,
        inputs,
        conf_frame: Atoms,
    ) -> Tuple[str, Path]:
        task_name = fp_task_pattern % idx
        task_path = Path(task_name)
        with set_directory(task_path):
            self.prep_task(conf_frame, inputs)
        return task_name, task_path

    def prep_task(
        self, 
        conf_frame: Atoms, inputs: Any):
        write(foo_conf_name, conf_frame)
        
        
class RunFoo(RunFp):
    def input_files(self) -> List[str]:
        """The input files to run a foo FP task

        Returns:
            List[str]: the list of input file names
        """
        return [foo_conf_name] 
    
    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a foo FP task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []
    
    
    
    
    def run_task(
        self,
        command: str,
        out:str= foo_conf_name,
        log:str = foo_log_name    
                ):
        log_name=log
        out_name = out
        atoms= read(foo_conf_name,index=0)
        energy = 1
        forces = np.random.rand(len(atoms),3)
        # delete calculator fields
        atoms.calc.results.clear()
        atoms.info['energy'] = energy
        atoms.set_array('force', forces)
        write(out_name,atoms)
        with open(log_name, "w") as log_file:
            log_file.write(f"THIS is a test: {command}\n")
        return out_name, log_name
        
        
    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """
        return [
            Argument(
                name="command",
                dtype=str,
                optional=True,
                default="foo",
                doc="Command to run the foo task.",
            ),
        ]

class FooInputs:
    def __init__(self, **kwargs):
        self.foo = "foo"
    @staticmethod
    def args():
        return []