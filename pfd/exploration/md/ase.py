from bz2 import compress
import json
from math import log
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict, field

import ase
import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from pfd.constants import ase_log_name, ase_traj_name

@dataclass
class MDParameters:
    """MD simulation parameters that can be serialized."""
    temp: float = 300.0  # K
    press: Optional[float] = None  # Bar (None for NVT)
    dt: float = 2.0  # fs
    nsteps: int = 30000  # NVT production steps
    traj_freq: int = 100  # frames
    log_freq: int = 100  # steps
    tau_t: float = 100.0  # damping factor * timestep
    tau_p: float = 1000  # damping factor for pressure (NPT)
    compressibility: float = 4.5e-5  # 1/bar (NPT)
    ensemble: str = "nvt"  # "nvt", "npt", or "both"
    custom_config: Dict[str, Any] = field(default_factory=dict)   # Custom configuration
    output_prefix: str = "md"
    
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MDParameters':
        return cls(**json.loads(json_str))
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'MDParameters':
        with open(filename, 'r') as f:
            return cls.from_json(f.read())
        

class MDRunner(Atoms):
    """
    ASE Atoms subclass that can run MD simulations from JSON parameters.
    
    Examples
    --------
    >>> # Create from structure file
    >>> md_atoms = MDRunner.from_file("structure.cif")
    >>> md_atoms.set_calculator(calculator)
    >>> 
    >>> # Run MD with parameters from JSON
    >>> md_atoms.run_md_from_json("md_params.json")
    >>> 
    >>> # Or run with direct parameters
    >>> params = MDParameters(temperature=500.0, nsteps_nvt=50000)
    >>> md_atoms.run_md(params)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md_history = []
        self.logger = self._setup_logger()

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
    def from_file(cls, filename: Union[str, Path]) -> 'MDRunner':
        """Create MDRunner from structure file."""
        atoms = read(filename, index=0)
        return cls(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
    
    def initialize_velocities(self, temperature: float, seed: Optional[int] = None) -> None:
        """Initialize Maxwell-Boltzmann velocity distribution."""
        if seed is not None:
            np.random.seed(seed)
        MaxwellBoltzmannDistribution(self, temperature_K=temperature)
        self.logger.info(f"Initialized velocities at {temperature} K")
    
    def run_npt(self, 
                params: MDParameters,
                log_file: Optional[str] = ase_log_name,
                traj_file: Optional[str] = ase_traj_name
                ) -> None:
        """Run NPT simulation."""

        if params.press is None:
            raise ValueError("Pressure must be specified for NPT simulation")
        timestep = params.dt * units.fs
        
        # Initialize velocities
        self.initialize_velocities(params.temp)
        
        # Setup NPT dynamics
        dyn = NPTBerendsen(
            self,
            timestep,
            temperature_K=params.temp,
            pressure_au=params.press * units.bar,
            taut=params.tau_t * units.fs,
            taup=params.tau_p * units.fs,
            compressibility_au=params.compressibility/units.bar,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
            
        traj = Trajectory(traj_file, 'w', atoms=dyn.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)

        # Run NPT
        self.logger.info("#### Starting MD...")
        start_time = time.time()
        dyn.run(params.nsteps)
        elapsed = time.time() - start_time
        self.logger.info(f"#### MD simulation completed in {elapsed:.2f} s")
        
        # Store history
        self.md_history.append({
            'type': 'NPT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'pressure': params.press,
            'duration': elapsed
        })
    
    def run_nvt(
        self, 
        params: MDParameters,
        log_file: Optional[str] = ase_log_name,
        traj_file: Optional[str] = ase_traj_name
        ) -> None:
        """Run NVT simulation."""
        timestep = params.dt * units.fs
        tdamp = params.tau_t * timestep
        
        # Initialize velocities if not already done
        if not hasattr(self, '_velocities_initialized'):
            self.initialize_velocities(params.temp)
            self._velocities_initialized = True
        
        # Setup NVT dynamics
        dyn = NoseHooverChainNVT(
            self,
            timestep,
            temperature_K=params.temp,
            tdamp=tdamp,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
        
        traj = Trajectory(traj_file, 'w', atoms=dyn.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
        # Run NVT
        self.logger.info("#### Starting NVT simulation...")
        start_time = time.time()
        dyn.run(params.nsteps)
        elapsed = time.time() - start_time
        self.logger.info(f"#### NVT simulation completed in {elapsed:.2f} s")
        
        # Store history
        self.md_history.append({
            'type': 'NVT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'duration': elapsed
        })
    
    def run_md(self, params: MDParameters) -> None:
        """Run MD simulation based on ensemble parameter."""
        if not hasattr(self, 'calc') or self.calc is None:
            raise ValueError("Calculator must be set before running MD")
            
        self.logger.info(f"Starting MD simulation with {len(self)} atoms")
        self.logger.info(f"Ensemble: {params.ensemble.upper()}")
        self.logger.info(f"Temperature: {params.temp} K")
        
        total_start = time.time()
        
        if params.ensemble.lower() == "npt":
            self.run_npt(params)
        elif params.ensemble.lower() == "nvt":
            self.run_nvt(params)
        else:
            raise ValueError(f"Unknown ensemble: {params.ensemble}")
        
        total_elapsed = time.time() - total_start
        self.logger.info(f"#### Total MD simulation completed in {total_elapsed:.2f} s")
    
    def run_md_from_json(self, json_file: Union[str, Path]) -> None:
        """Run MD simulation using parameters from JSON file."""
        params = MDParameters.from_file(json_file)
        self.logger.info(f"Loaded MD parameters from {json_file}")
        self.run_md(params)
    
    def run_md_from_dict(self, params_dict: Dict[str, Any]) -> None:
        """Run MD simulation using parameters from dictionary."""
        params = MDParameters(**params_dict)
        self.run_md(params)
    
    def get_md_summary(self) -> Dict[str, Any]:
        """Get summary of completed MD runs."""
        return {
            'total_runs': len(self.md_history),
            'runs': self.md_history,
            'total_steps': sum(run['steps'] for run in self.md_history),
            'total_duration': sum(run['duration'] for run in self.md_history)
        }

