import json
import logging
import os
import re
from pathlib import (
    Path,
)
from typing import (
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
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    OPIOSign,
    TransientError,
)

from ase import Atoms
from ase.io import read, write
import dpdata

from pfd.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
    fp_task_pattern
)
from pfd.exploration import task
from pfd.utils import (
    run_command,
    sort_atoms_by_atomic_number,
    get_element_types_from_sorted_atoms,
    dpdata2ase,
    set_directory
)

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

# global static variables
vasp_conf_name = "POSCAR"
vasp_input_name = "INCAR"
vasp_pot_name = "POTCAR"
#vasp_kp_name = "KPOINTS"


def clean_lines(string_list, remove_empty_lines=True):
    """[migrated from pymatgen]
    Strips whitespace, carriage returns and empty lines from a list of strings.

    Args:
        string_list: List of strings
        remove_empty_lines: Set to True to skip lines which are empty after
            stripping.

    Returns:
        List of clean strings with no whitespaces.
    """
    for s in string_list:
        clean_s = s
        if "#" in s:
            ind = s.index("#")
            clean_s = s[:ind]
        clean_s = clean_s.strip()
        if (not remove_empty_lines) or clean_s != "":
            yield clean_s


def loads_incar(incar: str):
    """[migrated from dpgen2]
    Parse a VASP INCAR string into a dictionary.
    Args:
        incar (str): INCAR string.

    Returns:
        dict: A dictionary representation of the INCAR parameters.
    """
    lines = list(clean_lines(incar.splitlines()))
    params = {}
    for line in lines:
        for sline in line.split(";"):
            m = re.match(r"(\w+)\s*=\s*(.*)", sline.strip())
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                params[key.upper()] = val
    return params


def dumps_incar(params: dict):
    incar = "\n".join([key + " = " + str(val) for key, val in params.items()]) + "\n"
    return incar


class PrepVasp(PrepFp):
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
        conf_frame: Atoms,
        vasp_inputs: VaspInputs,
    ):
        r"""Define how one Vasp task is prepared.

        Parameters
        ----------
        conf_frame : ase.Atoms
            One frame of configuration in the ase.Atoms format.
        vasp_inputs : VaspInputs
            The VaspInputs object handels all other input files of the task.
        """
        conf_frame= sort_atoms_by_atomic_number(conf_frame)
        atom_names = get_element_types_from_sorted_atoms(conf_frame)
        write(vasp_conf_name, conf_frame, format="vasp")
        incar = vasp_inputs.incar_template
        init_magmoms = conf_frame.get_initial_magnetic_moments()
        # check whether init_magmoms is np.zeros
        incar_dict = loads_incar(incar)
        #if vasp_inputs.kgamma
        incar_dict["KGAMMA"] = ".TRUE." if vasp_inputs.kgamma else ".FALSE."
        incar_dict["KSPACING"] = vasp_inputs.kspacing
        if np.any(init_magmoms):
            incar_dict["MAGMOM"] = " ".join([str(x) for x in init_magmoms])
        incar = dumps_incar(incar_dict)
        Path(vasp_input_name).write_text(incar)
        Path(vasp_pot_name).write_text(vasp_inputs.make_potcar(atom_names))

class RunVasp(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a vasp task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [vasp_conf_name, vasp_input_name, vasp_pot_name]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a vasp task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return ["job.json"]


    def run_task(
        self,
        command: str,
        out: str,
        log: str,
        **kwargs
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running vasp task
        out : str
            The name of the output data file.
        log : str
            The name of the log file

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """

        log_name = log
        out_name = out
        # run vasp
        command = " ".join([command, ">", log_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            logging.error(
                "".join(
                    ("vasp failed\n", "out msg: ", out, "\n", "err msg: ", err, "\n")
                )
            )
            raise TransientError("vasp failed")
        # dpdata as a lightweight parser
        sys = dpdata.LabeledSystem("OUTCAR")
        sys = dpdata2ase(sys)
        # convert system to Atoms format
        write(out_name,sys,format="extxyz")
        return out_name, log_name

    @staticmethod
    def args():
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_vasp_cmd = "The command of VASP"
        doc_vasp_log = "The log file name of VASP"
        doc_vasp_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default="vasp", doc=doc_vasp_cmd),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_vasp_out,
            ),
            Argument(
                "log", str, optional=True, default=fp_default_log_name, doc=doc_vasp_log
            ),
        ]