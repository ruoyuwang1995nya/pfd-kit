from dargs import Argument

from .task import (
    ExplorationTask,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)

from pfd.constants import (
    ase_conf_name,
    ase_input_name,
)
from typing import List, Optional, Dict

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
        temps: List[float],  # temperature
        press: Optional[List[float]] = None,
        ens: str = "npt",
        dt: float = 2,  # time step
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
        self, conf: str, temp: float, press: Optional[float]
    ) -> ExplorationTask:
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
        task = ExplorationTask()
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

    @classmethod
    def normalize_config(cls, data: Dict = {}, strict: bool = False) -> Dict:
        r"""Normalized the argument.

        Parameters
        ----------
        data : Dict
            The input dict of arguments.
        strict : bool
            Strictly check the arguments.

        Returns
        -------
        data: Dict
            The normalized arguments.

        """
        ta = cls.args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data

    @classmethod
    def args(cls):
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_temps = "List of temperatures for MD simulation."
        doc_press = "List of pressures for MD simulation (optional)."
        doc_ens = "Ensemble type (e.g., 'npt', 'nvt')."
        doc_dt = "MD time step (fs or ps)."
        doc_nsteps = "Number of MD steps."
        doc_trj_freq = "Trajectory output frequency."
        doc_tau_t = "Thermostat time constant."
        doc_tau_p = "Barostat time constant."
        doc_no_pbc = "Disable periodic boundary conditions."

        return [
            Argument("temps", list, optional=False, doc=doc_temps),
            Argument("press", list, optional=True, default=None, doc=doc_press),
            Argument("ens", str, optional=True, default="npt", doc=doc_ens),
            Argument("dt", float, optional=True, default=2, doc=doc_dt),
            Argument("nsteps", int, optional=True, default=1000, doc=doc_nsteps),
            Argument("trj_freq", int, optional=True, default=100, doc=doc_trj_freq),
            Argument("tau_t", float, optional=True, default=100, doc=doc_tau_t),
            Argument("tau_p", float, optional=True, default=500, doc=doc_tau_p),
            Argument("no_pbc", bool, optional=True, default=False, doc=doc_no_pbc),
        ]

    @classmethod
    def make_task_grp(
        cls,
        atom_ls_strs: List[str],
        config: Dict,
        n_sample: int = 1,
    ):
        # task["model_name_pattern"] = pytorch_model_name_pattern
        task_grp = AseTaskGroup()
        task_grp.set_md(**AseTaskGroup.normalize_config(config, strict=False))
        task_grp.set_conf(conf_list=atom_ls_strs, n_sample=n_sample, random_sample=True)
        return task_grp

    @classmethod
    def make_task_grp_from_conf(
        cls, task_grp_config: Dict, init_confs: List[str], *args, **kwargs
    ) -> "List[AseTaskGroup]":
        """
        Create ASE task group from configuration files and task group config.

        Parameters
        ----------
        init_confs : List[str]
            List of paths to initial configuration files
        task_grp_config : Dict
            Task group configuration containing conf_idx, n_sample, and other params

        Returns
        -------
        AseTaskGroup
            Configured ASE task group
        """
        from io import StringIO
        from ase.io import read, write

        confs_idx = task_grp_config.pop("conf_idx")
        n_sample = task_grp_config.pop("n_sample")
        # get structure in string format
        task_grp_ls = []
        for ii in confs_idx:
            atoms_ls_str = []
            atoms_ls = read(init_confs[ii], index=":")
            if not isinstance(atoms_ls, list):
                atoms_ls = [atoms_ls]
            for atoms in atoms_ls:
                buf = StringIO()
                write(buf, atoms, format="extxyz")
                atoms_ls_str.append(buf.getvalue())
            task_grp_ls.append(
                cls.make_task_grp(atoms_ls_str, task_grp_config, n_sample=n_sample)
            )
        return task_grp_ls
