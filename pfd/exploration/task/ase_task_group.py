
from math import tau
from .task import (
    ExplorationTask,
)
from .task_group import (
    ExplorationTaskGroup,
)
from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .ase import (
    ASEInput
)
from pfd.constants import (
    ase_conf_name,
    ase_input_name,
)
from typing import List, Optional

import itertools

from pfd.exploration.md import MDParameters

class AseTaskGroup(ConfSamplingTaskGroup):
    """
    AseTaskGroup is a task group for ASE-based tasks.

    It inherits from ExplorationTaskGroup and provides methods to set up
    ASE tasks with specific configurations.

    Attributes
    ----------
    ase_set : bool
        Indicates whether the ASE task settings have been configured.

    Methods
    -------
    set_ase_task(
        calc,
        model_name,
        model_path,
        n_sample=None,
        random_sample=True,
        task_name="ase-dp",
    ):
        Configure the ASE task with the provided parameters.

    make_task() -> ExplorationTaskGroup:
        Create and return an ExplorationTaskGroup containing the configured ASE task.
    """

    def __init__(self):
        super().__init__()
        self.ase_set = False

    def set_md(
        self,
        temps: List[float], # temperature
        press: Optional[List[float]] = None,
        ens: str = "npt",
        dt: float = 2, # time step
        nsteps: int = 1000,
        trj_freq: int = 100,
        tau_t: float = 100,
        tau_p: float = 500,
        no_pbc: bool = False,
    ):
        """_summary_

        Args:
            temps (List[float]): _description_
            ens (str, optional): _description_. Defaults to "npt".
            dt (float, optional): _description_. Defaults to 0.001.
            trj_freq (int, optional): _description_. Defaults to 10.
            tau_t (float, optional): _description_. Defaults to 0.1.
            tau_p (float, optional): _description_. Defaults to 0.5.
            pka_e (Optional[float], optional): _description_. Defaults to None.
            neidelay (Optional[int], optional): _description_. Defaults to None.
            no_pbc (bool, optional): _description_. Defaults to False.
        """
        self.temps = temps
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p
        self.no_pbc = no_pbc
        self.md_set = True

    def make_task(self) -> "AseTaskGroup":
        """
        Make the ASE task group.

        Returns
        -------
        task_grp: ExplorationTaskGroup
            Return one ASE task group.
        """
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.md_set:
            raise ValueError("ASE task is not set up")
        self.clear()
        confs = self._sample_confs()
        for cc, tt, pp in itertools.product(confs, self.temps, self.press):  # type: ignore
            self.add_task(self._make_ase_task(cc, tt, pp))
        return self
    
    
    def _make_ase_task(
        self, 
        conf: str, 
        temp: float, 
        press: Optional[float]) -> ExplorationTask:
        """
        Create an ASE task with the given configuration, temperature, and pressure.

        Parameters
        ----------
        conf : str
            The configuration string for the ASE task.
        temp : float
            The temperature for the ASE task.
        press : Optional[float]
            The pressure for the ASE task, if applicable.

        Returns
        -------
        ExplorationTask
            An instance of ExplorationTask configured for ASE.
        """
        task= ExplorationTask()
        task.add_file(ase_conf_name, conf)
        ase_input = MDParameters(
            temp=temp,
            press=press,
            ensemble=self.ens,
            dt=self.dt,
            nsteps=self.nsteps,
            traj_freq=self.trj_freq,
            log_freq=self.trj_freq,
            tau_t=self.tau_t,
            tau_p=self.tau_p,
        )
        task.add_file(ase_input_name, ase_input.to_json())
        return task