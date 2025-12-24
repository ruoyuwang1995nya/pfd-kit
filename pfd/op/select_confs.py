from pathlib import (
    Path,
)
from typing import (
    List,
    Dict,
    Optional
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    Parameter
)
from ase.io import read, write
from ase import Atoms
import numpy as np

from pfd.exploration.selector import (
    ConfSelector,
)
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SelectConfs(OP):
    """Select configurations from exploration trajectories for labeling."""
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "conf_selector": ConfSelector,
                "confs": Artifact(List[Path]),
                "init_confs": Artifact(List[Path], optional=True),
                "iter_confs": Artifact(Path, optional=True),  # previous selected configurations
                "optional_parameters": Parameter(Dict, default={}),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "confs": Artifact(Path),
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

            - `conf_selector`: (`ConfSelector`) Configuration selector.
            - `confs`: (`List[str]`) The exploration trajectories.
            - `init_confs`: (`Artifact(List[Path])`) The initial configurations.
            - `pre_confs`: (`Artifact(List[Path])`) The trajectories generated in the exploration.
            - `optional_parameters`: (`Dict`) The optional parameters

        Returns
        -------
        Any
            Output dict with components:
            - `confa`: (`Artifact(Path)`) The selected configurations.

        """
        optional_parameters = ip["optional_parameters"]
        conf_selector = ip["conf_selector"]
        trajs = self._expand_directories(ip["confs"])
        logger.info(f"Expanded {len(ip['confs'])} input paths to {len(trajs)} trajectory files.")
        ref_confs = []
        if init_confs:= ip.get("init_confs"):
            for t in init_confs:
                ref_confs.extend(read(t, index=":"))
        if iter_confs := ip.get("iter_confs"):
            ref_confs.extend(read(iter_confs, index=":"))
        confs = conf_selector.select(
            trajs,
        )
        logger.info(f"Select {len(confs)} configurations from trajectories.")
        
        
        
        max_sel = optional_parameters.get("max_sel")
        h_filter = optional_parameters.get("h_filter")
        if h_filter:
            confs = self.filter_by_entropy(
                confs,
                reference=ref_confs,
                max_sel=max_sel, 
                **h_filter)
        
        elif len(confs) > max_sel:
            logger.info(f"Selected {len(confs)} configurations, but max_sel is {max_sel}. Randomly select {max_sel} configurations.")
            sel_indices = np.random.choice(len(confs), max_sel, replace=False)
            confs = [confs[i] for i in sel_indices]
        
        out_path = Path("confs")
        out_path.mkdir(exist_ok=True)
        
        write(out_path / "confs.extxyz", confs, format="extxyz")
        return OPIO(
            {
                "confs": out_path / "confs.extxyz",
            }
        )
        
    def filter_by_entropy(
        self,
        iter_confs: List[Atoms],
        reference: List[Atoms]=[],
        chunk_size: int = 10,
        k=32,
        cutoff=5.0,
        batch_size: int = 1000,
        h = 0.015,
        max_sel: int =100,
        **kwargs
        )-> List[Atoms]:
        """Iteratively select configurations for maximum entropy."""

        try:
            import torch
            logger.info("Using torch entropy calculation")
            return _h_filter_gpu(
                iter_confs,
                reference,
                chunk_size=chunk_size,
                max_sel=max_sel,
                k=k,
                cutoff=cutoff,
                batch_size=batch_size,
                h=h,
                **kwargs
            )

        except ImportError:
            logger.info("Using CPU entropy (torch not available)")
            return _h_filter_cpu(
                iter_confs,
                reference,
                chunk_size=chunk_size,
                max_sel=max_sel,
                k=k,
                cutoff=cutoff,
                batch_size=batch_size,
                h=h,
                **kwargs
                )
        
    def _expand_directories(self, paths: List[Path]) -> List[Path]:
        """Expand directories to include all trajectory files within them.
        
        Parameters
        ----------
        paths : List[Path]
            List of file or directory paths
            
        Returns
        -------
        List[Path]
            Expanded list of file paths, with directories recursively searched for trajectory files
        """
        expanded_paths = []
        # Common trajectory file extensions
        traj_extensions = {'.traj', '.extxyz', '.xyz', '.dump', '.lammpstrj', '.nc', '.h5'}
        
        for path in paths:
            if path.is_file():
                expanded_paths.append(path)
            elif path.is_dir():
                # Recursively find all trajectory files in the directory
                for file_path in path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in traj_extensions:
                        expanded_paths.append(file_path)
                        logger.info(f"Found trajectory file: {file_path}")
            else:
                logger.warning(f"Path does not exist: {path}")
        
        return expanded_paths
        

def _h_filter_cpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    from quests.descriptor import get_descriptors
    from quests.entropy import entropy,delta_entropy
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size
        
    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype)
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)

    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end

    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = np.vstack(re_desc)
        delta = delta_entropy(x, dset_desc, h=h,batch_size=batch_size)
        num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
        atom_indices = []
        start = 0
        for n in num_atoms_per_structure:
            end = start + n
            atom_indices.append((start, end))
            start = end
        delta_sums = [delta[start:end].sum() for start, end in atom_indices]
        sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
        sorted_re_indices = [idx for idx, _ in sorted_pairs]
        selected_indices = sorted_re_indices[:chunk_size]
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        H = entropy(dset_desc, h=h, batch_size=batch_size)
        logger.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
    return dset_confs[num_ref:] # return only the newly selected ones

def _h_filter_gpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    import torch
    from quests.descriptor import get_descriptors
    from quests.gpu.entropy import delta_entropy,entropy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device} for GPU entropy calculation")
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size
    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype,concat=False)
    logger.info(f"Reading descriptors for iter_confs: {len(iter_desc)}")
    
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)
    logger.info(f"Reading descriptors for dset_confs: {len(dset_desc)}")

    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end
    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = torch.tensor(np.vstack(re_desc),device=device, dtype=torch.float32)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        delta = delta_entropy(x, y, h=h,batch_size=batch_size, device=device,**kwargs)
        delta = delta.cpu().numpy()
        num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
        atom_indices = []
        start = 0
        for n in num_atoms_per_structure:
            end = start + n
            atom_indices.append((start, end))
            start = end
        delta_sums = [delta[start:end].sum() for start, end in atom_indices]
        sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
        sorted_re_indices = [idx for idx, _ in sorted_pairs]
        selected_indices = sorted_re_indices[:chunk_size]
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        H = entropy(y, h=h, batch_size=batch_size,device=device)
        logger.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
    return dset_confs[num_ref:] # return only the newly selected ones