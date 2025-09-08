from pathlib import Path
from typing import Optional, Union, List
from dflow.python import FatalError
from pfd.op.train.train import Train
from dargs import Argument
from ase.io import read
from ase.units import GPa
import numpy as np
import random
import logging
import os
import json

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"), 
        ],
)
logger = logging.getLogger(__name__)

class MatterSim(Train):
    """Train OP for fine-tuning MatterSim models."""

    train_script_name = "config.json"
    model_file = "last_model.pth"
    log_file = "train.log"
    lcurve_file = "lcurve.out"


    def run_train(
        self,
        config: dict,
        init_model: Optional[Union[str, Path]] = None,
        init_data: Optional[List[Path]] = None,
        iter_data: Optional[Path] = None,
        valid_data: Optional[Path] = None,
        #optional_files: Optional[List[Path]]=None,
        **kwargs,

    ):
        """Run the MatterSim fine-tuning process."""
        if (not init_data or len(init_data) == 0) and not iter_data:
            raise FatalError(
                "At least one data source must be provided: "
                "either 'init_data' (non-empty list) or 'iter_data' (valid path)"
            )
            
        finetune=config.pop("finetune_mode",False)
        config = MatterSim.normalize_config(data=config)
        ## possible implement of fine-tuning
        
        ## import dependence within the OP.execute()
        from mattersim.datasets.utils.build import build_dataloader
        from mattersim.forcefield.m3gnet.scaling import AtomScaling
        from mattersim.forcefield.potential import Potential
        import torch
        
        # if not set than set to default single process
        os.environ.setdefault("LOCAL_RANK", "0")  # Local rank of the process
        os.environ.setdefault("RANK", "0")        # Global rank of the process
        os.environ.setdefault("WORLD_SIZE", "1")  # Total number of processes
        os.environ.setdefault("MASTER_ADDR", "localhost")  # Address of the master node
        os.environ.setdefault("MASTER_PORT", "29500")  # Port of the master node

        if config["device"] == "cuda":
            torch.cuda.empty_cache()
            torch.distributed.init_process_group(backend="nccl")
        else:
            torch.distributed.init_process_group(backend="gloo")
        torch.distributed.barrier()

        # set random seed
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        
        if config["device"] == "cuda":
            torch.cuda.set_device(0)

        # write data to a single extxyz file
        atoms_train=[]
        if init_data and len(init_data) > 0:
            for ii in init_data:
                atoms_train += read(ii,index=":")
        if iter_data:
            atoms_train += read(iter_data, index=":")
            
        energies = []
        forces = [] if config["include_forces"] else None
        stresses = [] if config["include_stresses"] else None
        logger.info("Processing training datasets...")
        for atoms in atoms_train:
            energies.append(atoms.get_potential_energy())
            if config["include_forces"]:
                forces.append(atoms.get_forces())
        if config["include_stresses"]:
            stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa

        dataloader = build_dataloader(
            atoms_train,
            energies,
            forces,
            stresses,
            shuffle=True,
            pin_memory=(config["device"] == "cuda"),
            is_distributed=True,
            **config
        )

        if config["re_normalize"]:
            scale = AtomScaling(
                atoms=atoms_train,
                total_energy=energies,
                forces=forces,
                verbose=True,
                **config,
                ).to(config["device"])

        val_dataloader = None
        potential = Potential.from_checkpoint(
            load_path=str(init_model),
            load_training_state=False,
            **config,
        )

        if config["re_normalize"]:
            potential.model.set_normalizer(scale)

        if config["device"] == "cuda":
            potential.model = torch.nn.parallel.DistributedDataParallel(potential.model)
        torch.distributed.barrier()

        potential.train_model(
            dataloader,
            val_dataloader,
            loss=torch.nn.HuberLoss(delta=0.01),
            is_distributed=True,
            **config,
    )
        with open(self.train_script_name, "w") as fp:
            json.dump(config, fp, indent=4
                      )
        with open(self.lcurve_file,"w") as fp:
            fp.write("based")

    @staticmethod
    def training_args():
        """Return the training arguments for MatterSim."""
        return [
        # Path parameters
        Argument("run_name", str, default="train", optional=True, doc="Name of the run."),
        Argument("save_path", str, default="./",optional=True, doc="Path to save the model."),
        Argument("save_checkpoint", bool, default=True, optional=True,doc="Save checkpoint during training."),
        Argument("ckpt_interval", int, default=10, optional=True,doc="Save checkpoint every ckpt_interval epochs."),
        Argument("device", str, default="cuda", optional=True,doc="Device to use for training (e.g., 'cuda' or 'cpu')."),

        # Model parameters
        Argument("cutoff", float, default=5.0, optional=True,doc="Cutoff radius for two-body interactions."),
        Argument("threebody_cutoff", float, default=4.0, optional=True,doc="Cutoff radius for three-body interactions."),

        # Training parameters
        Argument("epochs", int, default=10,optional=True, doc="Number of training epochs."),
        Argument("batch_size", int, default=8, optional=True,doc="Batch size for training."),
        Argument("lr", float, default=2e-4, optional=True,doc="Learning rate."),
        Argument("step_size", int, default=10, optional=True,doc="Step epoch for learning rate scheduler."),
        Argument("include_forces", bool, default=True, optional=True,doc="Include forces in training."),
        Argument("include_stresses", bool, default=False, optional=True,doc="Include stresses in training."),
        Argument("force_loss_ratio", float, default=1.0, optional=True,doc="Weight for force loss."),
        Argument("stress_loss_ratio", float, default=0.1,optional=True, doc="Weight for stress loss."),
        Argument("early_stop_patience", int, default=10,optional=True, doc="Patience for early stopping."),
        Argument("seed", int, default=42,optional=True, doc="Random seed for reproducibility."),

        # Scaling parameters
        Argument("re_normalize", bool, default=False, optional=True, doc="Re-normalize energy and forces."),
        Argument("scale_key", str, default="per_species_forces_rms", optional=True, doc="Key for scaling forces."),
        Argument("shift_key", str, default="per_species_energy_mean_linear_reg", optional=True, doc="Key for shifting energy."),
        Argument("init_scale", float, default=None, optional=True, doc="Initial scale value."),
        Argument("init_shift", float, default=None, optional=True, doc="Initial shift value."),
        Argument("trainable_scale", bool, default=False, optional=True, doc="Allow scale to be trainable."),
        Argument("trainable_shift", bool, default=False, optional=True, doc="Allow shift to be trainable."),

    ]

    @staticmethod
    def normalize_config(data={}):
        """Normalize the training configuration."""
        ta = MatterSim.training_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data