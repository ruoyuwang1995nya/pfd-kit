
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple
)
import os
import dflow

from pfd.utils import (
    bohrium_config_from_dict,
    workflow_config_from_dict,
    perturb
)

from ase import Atoms
from ase.io import read,write


def global_config_workflow(
    wf_config,
):
    # dflow_config, dflow_s3_config
    workflow_config_from_dict(wf_config)

    if os.getenv("DFLOW_DEBUG"):
        dflow.config["mode"] = "debug"
        return None

    # bohrium configuration
    if wf_config.get("bohrium_config") is not None:
        bohrium_config_from_dict(wf_config["bohrium_config"])


def expand_sys_str(root_dir: Union[str, Path]) -> List[str]:
    root_dir = Path(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches


def expand_idx(in_list) -> List[int]:
    ret = []
    for ii in in_list:
        if isinstance(ii, int):
            ret.append(ii)
        elif isinstance(ii, str):
            # e.g., 0-41:1
            step_str = ii.split(":")
            if len(step_str) > 1:
                step = int(step_str[1])
            else:
                step = 1
            range_str = step_str[0].split("-")
            if len(range_str) == 2:
                ret += range(int(range_str[0]), int(range_str[1]), step)
            elif len(range_str) == 1:
                ret += [int(range_str[0])]
            else:
                raise RuntimeError("not expected range string", step_str[0])
    ret = sorted(list(set(ret)))
    return ret


def perturb_cli(
    atoms_path_ls: List[Union[str,Path]], 
    pert_num: int, 
    cell_pert_fraction: float, 
    atom_pert_distance: float, 
    atom_pert_style: str, 
    atom_pert_prob: float, 
    supercell: Optional[Union[int, Tuple[int,int,int]]] = None,
    ):
    """A CLI function to perturb structures from file paths.
    """
    #pert_atoms_ls = []
    for atoms_path in atoms_path_ls:
        atoms_ls = read(atoms_path,index=':')
        pert_atom_ls = perturb(
            atoms_ls,
            pert_num,
            cell_pert_fraction,
            atom_pert_distance,
            atom_pert_style,
            atom_pert_prob=atom_pert_prob,
            supercell=supercell
        )
        write("pert_"+Path(atoms_path).stem+'.extxyz',pert_atom_ls,format='extxyz')
