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
                "pre_confs": Artifact(Path, optional=True),  # previous selected configurations
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
            - `type_map`: (`List[str]`) The type map.
            - `trajs`: (`Artifact(List[Path])`) The trajectories generated in the exploration.
            - `model_devis`: (`Artifact(List[Path])`) The file storing the model deviation of the trajectory. The order of model deviation storage is consistent with that of the trajectories. The order of frames of one model deviation storage is also consistent with tat of the corresponding trajectory.

        Returns
        -------
        Any
            Output dict with components:
            - `report`: (`ExplorationReport`) The report on the exploration.
            - `conf`: (`Artifact(List[Path])`) The selected configurations.

        """
        optional_parameters = ip["optional_parameters"]
        conf_selector = ip["conf_selector"]
        trajs = ip["confs"]
        pre_confs = ip.get("pre_confs")
        
        confs = conf_selector.select(
            trajs,
        )
        logger.info(f"Select {len(confs)} configurations from trajectories.")
        # entropy based selection
        h_filter = optional_parameters.get("h_filter")
        if h_filter:
            confs = self.filter_by_entropy(confs,referece=pre_confs, **h_filter)
        
        out_path = Path("confs")
        out_path.mkdir(exist_ok=True)
        max_sel = optional_parameters.get("max_sel")
        if max_sel:
            if len(confs) > max_sel:
                logger.info(f"Selected {len(confs)} configurations, but max_sel is {max_sel}. Randomly select {max_sel} configurations.")
                sel_indices = np.random.choice(len(confs), max_sel, replace=False)
                confs = [confs[i] for i in sel_indices]
        
        write(out_path / "confs.extxyz", confs, format="extxyz")
        return OPIO(
            {
                "confs": out_path / "confs.extxyz",
            }
        )
        
    def filter_by_entropy(
        self,
        confs: List[Atoms],
        reference: Optional[Path]=None,
        k=32,
        cutoff=5.0,
        batch_size: int = 1000,
        h = 0.015,
        entropy_threshold: float = 0.01,
        **kwargs
        )-> List[Atoms]:
        """Filter structures to maximize entropy/diversity."""
        from quests.descriptor import get_descriptors
        from tqdm import tqdm
        def create_entropy_function():
            """Factory function to create the appropriate entropy function."""
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                from quests.gpu.entropy import entropy as gpu_entropy
        
                def get_entropy(x: np.ndarray, **kwargs):
                    x_tensor = torch.from_numpy(x)
                    return gpu_entropy(x_tensor, device=device, **kwargs)
            
                logger.info(f"Using GPU entropy with device: {device}")
                return get_entropy
        
            except ImportError:
                from quests.entropy import entropy as cpu_entropy
                def get_entropy(x: np.ndarray, **kwargs):
                    return cpu_entropy(x, **kwargs)
                logger.info("Using CPU entropy (torch not available)")
                return get_entropy
            
        num_confs=len(confs)
        get_entropy = create_entropy_function()
        filtered_structures = []
            
        if reference is not None:
            reference = read(reference, index=":")
        else:
            n_ref = max(1, min(100, len(confs) // 10))
            ref_indices = np.random.choice(len(confs), n_ref, replace=False)
            reference = [confs[i] for i in ref_indices]
            write('text.extxyz', reference, format='extxyz')
            other_indices = np.setdiff1d(np.arange(len(confs)), ref_indices)
            confs = [confs[i] for i in other_indices]
        current_descriptors = get_descriptors(reference,k=k,cutoff=cutoff)
        for atoms in tqdm(confs):
            cand_desc = get_descriptors([atoms],k=k,cutoff=cutoff)
            current_entropy = get_entropy(
                current_descriptors, 
                h=h,
                batch_size=batch_size,
            )
            tmp_descriptors = np.vstack([current_descriptors, cand_desc])
            entropy_tmp = get_entropy(
                tmp_descriptors, 
                h=0.015, batch_size=10000
            )
            entropy_delta = entropy_tmp - current_entropy
            if entropy_delta > entropy_threshold:
                filtered_structures.append(atoms)
                current_descriptors = tmp_descriptors
        logger.info(f"Entropy filtering: selected {len(filtered_structures)} structures from {num_confs} candidates.")
        logger.info(f"Entropy: {entropy_tmp}")
        return filtered_structures
