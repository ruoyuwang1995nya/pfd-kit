"""Implements replica exchange molecular dynamics using ASE."""
from typing import Optional, List, Tuple, TypedDict, Union
import time

from pathlib import Path
from dataclasses import dataclass, asdict

import json
import logging

import numpy as np
from numpy.random import Generator

from ase import units, Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from pfd.utils.optimize_struct_utils import relax_structure_ase
from pfd.constants import ase_log_name, ase_traj_name


# Only Langevin dynamics supported for now.
class ReplicaDict(TypedDict):
    init_id: int
    atoms: Atoms
    dyn: Langevin
    T: float


def initialize_replicas(
        atoms: Atoms,
        calculator: Calculator,
        min_temperature: float=300,
        max_temperature: float=1000,
        num_temperature_steps: int=5,
        timestep_fs: float=2.0,
        friction_fs: float=0.02,
        rng: Optional[Generator]=None,
        log_file: str = ase_log_name,
) -> List[ReplicaDict]:
    """Initialize replicas for temperature replica exchange MD.

    Create independent NVT (Langevin) replicas with fixed cell (slab-safe).
    Exchange will be done by swapping thermostat target temperatures.

    Args:
        atoms (Atoms): Initial structure for all replicas.
        calculator (Calculator): ASE calculator to use for all replicas.
        min_temperature (float): Minimum temperature in K. Defaults to 300 K.
        max_temperature (float): Maximum temperature in K. Defaults to 1000 K.
        num_temperature_steps (int): Number of temperature steps. Defaults to 5.
        timestep_fs (float): time step in femtoseconds. Defaults to 2.0 fs.
        friction_fs (float): Friction factor of ASE Langevin simulation in fs^-1. Defaults to 0.02 fs^-1.
        rng (Generator, optional): Random number generator. Defaults to None.
        log_file (str): Log file name for ASE MD output. Defaults to ase_log_name.
            Notice that each replica will have its own log file named. For example, if
            log_file="remd.log", then replica 0 will log to "remd.replica0.log".
    Returns:
        List[ReplicaDict]: List of replicas, each represented as a dictionary with keys:
            - "init_id": Initial replica index to keep track after swapping.
            - "atoms": Atoms object for the replica after last replica swap.
            - "dyn": Langevin dynamics object for the replica.
            - "T": Target temperature in K.
    """
    if rng is None:
        rng = np.random.default_rng()

    timestep = timestep_fs * units.fs
    replicas = []

    # IMPORTANT: slab safety -> keep cell fixed, do not use barostat
    fixed_cell = atoms.cell.copy()

    temperatures = np.linspace(
        min_temperature, max_temperature, num_temperature_steps
    ).tolist()
    for rid, T in enumerate(temperatures):
        a = atoms.copy()
        a.calc = calculator
        a.set_cell(fixed_cell, scale_atoms=False)

        # Initialize velocities from Maxwell distribution at this replica temperature
        MaxwellBoltzmannDistribution(a, temperature_K=T, rng=rng)

        logname_part1 = '.'.join(log_file.split('.')[:-1])
        logname_part2 = log_file.split('.')[-1]
        replica_log_file = f"{logname_part1}.replica{rid}.{logname_part2}"
        dyn = Langevin(
            a, timestep=timestep, temperature_K=T, friction=friction_fs / units.fs,
            logfile=replica_log_file, rng=rng,
        )

        # Initial_id tracks trajectory identification after swapping and re-sorting.
        replicas.append({"init_id": rid, "atoms": a, "dyn": dyn, "T": float(T)})
    return replicas


def attempt_temperature_swap(
        rep_i: ReplicaDict,
        rep_j: ReplicaDict,
        rng: Optional[Generator]=None
) -> bool:
    """Attempt a temperature swap between two replicas (i, j) in canonical T-REMD.

    Keep (q, p) unchanged; swap only target temperatures.
    Acceptance depends only on potential energies:
        acc = min(1, exp((β_i-β_j)*(U_i-U_j)))
    Args:
        rep_i (ReplicaDict): Replica i dictionary.
        rep_j (ReplicaDict): Replica j dictionary.
        rng (Generator, optional): Random number generator. Defaults to None.
    Returns:
        bool: True if swap accepted, False otherwise.
    """
    if rng is None:
        rng = np.random.default_rng()

    ai, aj = rep_i["atoms"], rep_j["atoms"]
    Ti, Tj = rep_i["T"], rep_j["T"]

    Ui = ai.get_potential_energy()
    Uj = aj.get_potential_energy()

    beta_i = 1.0 / (units.kB * Ti)
    beta_j = 1.0 / (units.kB * Tj)

    delta = (beta_i - beta_j) * (Ui - Uj)  # log acceptance ratio

    if rng.random() < np.exp(min(0.0, delta)):
        # Swap temperature labels
        rep_i["T"], rep_j["T"] = rep_j["T"], rep_i["T"]

        # Update thermostat target temperatures in the Langevin dynamics objects.
        # ASE Langevin stores temperature in different attribute names across versions;
        # we handle the common ones robustly.
        for rep in (rep_i, rep_j):
            dyn = rep["dyn"]
            Tnew = rep["T"]
            # Reset temperature in ASE Langevin
            dyn.set_temperature(temperature_K=Tnew)
        return True
    return False


def exchange_sweep_oddeven(
        replicas: List[ReplicaDict],
        cycle: int,
        rng: Optional[Generator]=None
) -> Tuple[int, int]:
    """Perform one exchange sweep using odd-even scheme.
    
    Args:
        replicas (List[ReplicaDict]): List of replicas.
        cycle (int): Current cycle number (1-based).
        rng (Generator, optional): Random number generator. Defaults to None.
    Returns:
        Tuple[int, int]: Number of accepted exchanges and number of attempted exchanges.
    """
    # Ensure that every replica participates at most 1 exchange per cycle.
    if rng is None:
        rng = np.random.default_rng()
    accepted = 0
    start = 0 if (cycle % 2 == 0) else 1
    sweep_range = list(range(start, len(replicas) - 1, 2))
    for i in sweep_range:
        if attempt_temperature_swap(replicas[i], replicas[i + 1], rng=rng):
            accepted += 1

    # Sort by temperature.
    replicas.sort(key=lambda rep: rep["T"])
    return accepted, len(sweep_range)


def sample_replica_exchange_langevin(
        atoms: Atoms,
        calculator: Calculator,
        min_temperature: float=300,
        max_temperature: float=1000,
        num_temperature_steps: int=5,
        timestep_fs: float=2.0,
        friction_fs: float=0.02,
        num_cycles: int=200,  # Total of 100 ps.
        md_steps_per_cycle: int=250, # 0.5 ps per swap attempt.
        log_every_cycles: int=10,
        drop_start_fraction: float=0.2,  # 20 ps allowed to equilibrate.
        num_samples: int=20,
        seed: Optional[int]=None,
        logger: Optional[logging.Logger]=None,
        log_file: str = ase_log_name,
) -> List[Atoms]:
    """Initialize and run temperature replica exchange MD.

    Does not relax the initial structure; assumes it is pre-relaxed.
    Args:
        atoms (Atoms): Initial structure for all replicas.
        calculator (Calculator): ASE calculator to use for all replicas.
        min_temperature (float): Minimum temperature in K. Defaults to 300 K.
        max_temperature (float): Maximum temperature in K. Defaults to 1000 K.
        num_temperature_steps (int): Number of temperature steps. Defaults to 5.
        timestep_fs (float): time step in femtoseconds. Defaults to 2.0 fs.
        friction_fs (float): Friction factor of ASE Langevin simulation in fs^-1.
         Defaults to 0.02 fs^-1.
        num_cycles (int): Number of exchange cycles. Defaults to 200.
        md_steps_per_cycle (int): Number of MD steps per exchange cycle. Defaults to 250.
        log_every_cycles (int): Logging frequency in cycles. Defaults to 10.
        drop_start_fraction (float): Fraction of initial cycles to drop for equilibration.
         Defaults to 0.2.
        num_samples (int): Number of structures to sample from the combined replicas.
         Defaults to 20.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        logger (logging.Logger, optional): Logger for logging messages. Defaults to None.
        log_file (str): Log file name for ASE MD output. Defaults to ase_log_name.
            Not dumped into logger, but passed to ASE Dynamics if needed.
            Notice that each replica will have its own log file named. For example, if
            log_file="remd.log", then replica 0 will log to "remd.replica0.log".
    Returns:
        List[Atoms]: List of sampled structures from the replicas.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    rng = np.random.default_rng(seed)
    print("Initialize replica sets.")
    replicas = initialize_replicas(
        atoms,
        calculator,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        num_temperature_steps=num_temperature_steps,
        timestep_fs=timestep_fs,
        friction_fs=friction_fs,
        rng=rng,
        log_file=log_file,
    )
    samples = []

    num_sample_cycles = int(np.ceil(num_samples / num_temperature_steps))
    sample_cycle_start = int(num_cycles * drop_start_fraction)
    delta_sample_cycles = max(1, (num_cycles - sample_cycle_start) // num_sample_cycles)
    # Evenly sample steps to minimize correlation between sampled structures.
    sample_cycles = list(range(sample_cycle_start + 1, num_cycles + 1, delta_sample_cycles))

    tot_accs = 0
    tot_ntrials = 0
    for cycle in range(1, num_cycles + 1):
        # 1) Propagate each replica independently
        for rep in replicas:
            rep["dyn"].run(md_steps_per_cycle)

        # 2) Attempt exchanges, then sort by increasing temperature.
        acc, ntrial = exchange_sweep_oddeven(replicas, cycle, rng=rng)
        tot_accs += acc
        tot_ntrials += ntrial

        if cycle % log_every_cycles == 0:
            Us = [rep["atoms"].get_potential_energy() for rep in replicas]
            ids = [rep["init_id"] for rep in replicas]
            logger.info(
                f"Cycle: {cycle:5d} | "
                f"acc ratio: {(tot_accs / tot_ntrials):.3f} | "
                f"U: {[float(u) for u in Us]} | "
                f"ids: {ids}"
            )

        if cycle in sample_cycles:
            samples.extend([rep["atoms"].copy() for rep in replicas])


    if len(samples) > num_samples:
        samples = [samples[idx] for idx in rng.choice(len(samples), size=num_samples, replace=False)]

    return samples


@dataclass
class REMDParameters:
    """Parameters for REMD simulation.

    Attributes:
        fmax (float): Force convergence criterion for initial relaxation.
        max_relax_steps (int): Maximum steps for initial relaxation.
        relax_is_slab (bool): Whether initial structure is slab/interface.
        min_temperature (float): Minimum temperature in K.
        max_temperature (float): Maximum temperature in K.
        num_temperature_steps (int): Number of temperature steps.
        timestep_fs (float): Time step in femtoseconds.
        friction_fs (float): Friction factor of ASE Langevin simulation in fs^-1.
        num_cycles (int): Number of exchange cycles.
        md_steps_per_cycle (int): Number of MD steps per exchange cycle.
        log_every_cycles (int): Logging frequency in cycles.
        drop_start_fraction (float): Fraction of initial cycles to drop for equilibration.
        num_samples (int): Number of structures to sample from the combined replicas.
        seed (Optional[int]): Random seed for reproducibility.
    """
    fmax: float = 0.01  # Force convergence criterion for initial relaxation
    max_relax_steps: int = 2000  # Max steps for initial relaxation
    relax_is_slab: bool = True  # Whether initial structure is slab/interface.
    # REMD parameters. Langevin dynamics only for now.
    min_temperature: float = 300.0
    max_temperature: float = 1000.0
    num_temperature_steps: int = 5
    timestep_fs: float = 2.0
    friction_fs: float = 0.02
    num_cycles: int = 200
    md_steps_per_cycle: int = 250
    log_every_cycles: int = 10
    drop_start_fraction: float = 0.2
    num_samples: int = 20
    seed: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> 'REMDParameters':
        return cls(**json.loads(json_str))

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'REMDParameters':
        with open(filename, 'r') as f:
            return cls.from_json(f.read())


class ReplicaRunner:
    """
    REMD simulation runner that wraps an ASE Atoms object.

    Examples
    --------
    >>> # Create from structure file
    >>> re_runner = ReplicaRunner.from_file("structure.cif")
    >>> re_runner.atoms.set_calculator(...)
    >>>
    >>> # Run MD with parameters from JSON
    >>> re_runner.run_md_from_json("md_params.json")
    >>>
    >>> # Or run with direct parameters
    >>> params = REMDParameters(temperature=500.0, nsteps_nvt=50000)
    >>> re_runner.run_md(params)
    """

    def __init__(self, atoms: Atoms):
        """Initialize MDRunner with an Atoms object."""
        self.atoms = atoms
        self.logger = self._setup_logger()

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atoms)

    def set_calculator(self, calc) -> None:
        """Set calculator for the atoms."""
        self.atoms.set_calculator(calc)

    @property
    def calc(self):
        """Get the calculator."""
        return self.atoms.calc

    @calc.setter
    def calc(self, calc):
        """Set the calculator."""
        self.atoms.calc = calc

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for MD simulation."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'ReplicaRunner':
        """Create MDRunner from structure file."""
        atoms_data = read(filename, index=0)
        # Ensure we have a single Atoms object, not a list
        if isinstance(atoms_data, list):
            atoms = atoms_data[0]
        else:
            atoms = atoms_data
        return cls(atoms)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> 'ReplicaRunner':
        """Create MDRunner from existing Atoms object."""
        return cls(atoms)

    def run_opt(
            self,
            params: REMDParameters,
            log_file: str = ase_log_name,
    ):
        """Run structure optimization with given parameters."""
        self.logger.info(f"#### Starting structure optimization. Is slab: {params.relax_is_slab}")
        start_time = time.time()
        relaxed_atoms = relax_structure_ase(
            atoms_init=self.atoms,
            calculator=self.calc,
            fmax_final=params.fmax,
            max_steps=params.max_relax_steps,
            is_slab=params.relax_is_slab,
            clear_calculator=False,  # Prevent calculator reset.
            logfile=log_file,
        )
        elapsed_time = time.time() - start_time
        self.atoms = relaxed_atoms  # Will still have the calculator attached.
        self.logger.info(f"#### Structure optimization completed in {elapsed_time:.2f} seconds.")


    def run_md(
            self,
            params: REMDParameters,
            log_file: str = ase_log_name,
            traj_file: str = ase_traj_name,
            pre_relax: bool = True,
    ):
        """Run REMD simulation with given parameters.

        Args:
            params (REMDParameters): Parameters for REMD simulation.
            log_file (str): Log file name. Defaults to ase_log_name.
            traj_file (str): Trajectory file name. Defaults to ase_traj_name.
            pre_relax (bool): Whether to perform initial structure relaxation. Defaults to True.
                Initial relaxation parameters will be taken from params.
                Often recommended to ensure a good starting structure.
        """
        if not hasattr(self.atoms, 'calc') or self.atoms.calc is None:
            raise ValueError("Calculator must be set before running REMD.")

        if pre_relax:
            self.run_opt(params, log_file=log_file)

        self.logger.info(f"Starting REMD simulation with {len(self.atoms)} atoms.")
        self.logger.info(f"Ensemble: NVT Langevin.")
        temperatures = np.linspace(
            params.min_temperature, params.max_temperature, params.num_temperature_steps
        ).tolist()
        self.logger.info(f"Temperatures: ({temperatures}) K")

        start_time = time.time()
        samples = sample_replica_exchange_langevin(
            atoms=self.atoms,
            calculator=self.calc,
            min_temperature=params.min_temperature,
            max_temperature=params.max_temperature,
            num_temperature_steps=params.num_temperature_steps,
            timestep_fs=params.timestep_fs,
            friction_fs=params.friction_fs,
            num_cycles=params.num_cycles,
            md_steps_per_cycle=params.md_steps_per_cycle,
            log_every_cycles=params.log_every_cycles,
            drop_start_fraction=params.drop_start_fraction,
            num_samples=params.num_samples,
            seed=params.seed,
            logger=self.logger,
            log_file=log_file,  # The ase output log file.
        )
        elapsed_time = time.time() - start_time
        self.logger.info(f"REMD simulation completed in {elapsed_time:.2f} seconds.")

        # Save all sampled frames into a trajectory file.
        with Trajectory(traj_file, mode="w") as traj:
            for sample in samples:
                traj.write(sample)

        # Concatenate replica log files to generate a main log file.
        logname_part1 = '.'.join(log_file.split('.')[:-1])
        logname_part2 = log_file.split('.')[-1]
        with open(log_file, 'w') as main_log:
            for rid in range(params.num_temperature_steps):
                replica_log_file = f"{logname_part1}.replica{rid}.{logname_part2}"
                main_log.write(f"{replica_log_file}:\n")
                with open(replica_log_file, 'r') as rep_log:
                    main_log.write(rep_log.read())

    def run_md_from_json(

            self,
            json_file: Union[str, Path],
            log_file: str = ase_log_name,
            traj_file: str = ase_traj_name,
    ):
        """Run REMD simulation with parameters from JSON file."""
        params = REMDParameters.from_file(json_file)
        self.run_md(params, log_file=log_file, traj_file=traj_file)

    def run_md_from_dict(
            self,
            params_dict: dict,
            log_file: str = ase_log_name,
            traj_file: str = ase_traj_name,
    ):
        """Run REMD simulation with parameters from dictionary."""
        params = REMDParameters(**params_dict)
        self.run_md(params, log_file=log_file, traj_file=traj_file)
