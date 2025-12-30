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
from pfd.exploration.md import REMDParameters


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
        # Add a parameter to specify the MD runner type.
        runner: str = "md",
        # Some common parameters used by all runners.
        no_pbc: bool = False,
        # MDRunnner's parameters. Many actually never appear here, amazing, huh?
        temps: Optional[List[float]] = None,  # temperature
        press: Optional[List[float]] = None,
        ens: str = "npt",
        dt: float = 2,  # time step
        nsteps: int = 1000,
        trj_freq: int = 100,
        tau_t: float = 100,
        tau_p: float = 500,
        # ReplicaRunner's parameters.
        fmax: float = 0.01,
        max_relax_steps: int = 2000,
        relax_is_slab: bool = True,
        min_temperature: float = 300.0,
        max_temperature: float = 1000.0,
        num_temperature_steps: int = 5,
        timestep_fs: float = 2.0,
        friction_fs: float = 0.02,
        num_cycles: int = 200,
        md_steps_per_cycle: int = 250,
        log_every_cycles: int = 10,
        drop_start_fraction: float = 0.2,
        num_samples: int = 20,
        seed: Optional[int] = None,
    ):
        """_summary_

        Args:
            runner (str, optional): _description_. Defaults to "md".
            no_pbc (bool, optional): _description_. Defaults to False.
            temps (List[float]): _description_
            press (Optional[List[float]], optional): _description_. Defaults to None.
            nsteps (int, optional): _description_. Defaults to 1000.
            ens (str, optional): _description_. Defaults to "npt".
            dt (float, optional): _description_. Defaults to 0.001.
            trj_freq (int, optional): _description_. Defaults to 10.
            tau_t (float, optional): _description_. Defaults to 0.1.
            tau_p (float, optional): _description_. Defaults to 0.5.
            fmax (float, optional): _description_. Defaults to 0.01.
            max_relax_steps (int, optional): _description_. Defaults to 2000.
            relax_is_slab (bool, optional): _description_. Defaults to True.
            min_temperature (float, optional): _description_. Defaults to 300.0.
            max_temperature (float, optional): _description_. Defaults to 1000.0.
            num_temperature_steps (int, optional): _description_. Defaults to 5.
            timestep_fs (float, optional): _description_. Defaults to 2.0.
            friction_fs (float, optional): _description_. Defaults to 0.02.
            num_cycles (int, optional): _description_. Defaults to 200.
            md_steps_per_cycle (int, optional): _description_. Defaults to 250.
            log_every_cycles (int, optional): _description_. Defaults to 10.
            drop_start_fraction (float, optional): _description_. Defaults to 0.2.
            num_samples (int, optional): _description_. Defaults to 20.
            seed (Optional[int], optional): _description_. Defaults to None.
        """
        self.runner = runner
        self.no_pbc = no_pbc
        self.temps = temps if temps is not None else [None]
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p

        self.fmax = fmax
        self.max_relax_steps = max_relax_steps
        self.relax_is_slab = relax_is_slab
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.num_temperature_steps = num_temperature_steps
        self.timestep_fs = timestep_fs
        self.friction_fs = friction_fs
        self.num_cycles = num_cycles
        self.md_steps_per_cycle = md_steps_per_cycle
        self.log_every_cycles = log_every_cycles
        self.drop_start_fraction = drop_start_fraction
        self.num_samples = num_samples
        self.seed = seed

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
        if self.runner == "md":
            for cc, tt, pp in itertools.product(confs, self.temps, self.press):  # type: ignore
                assert tt is not None, "Temperature must be specified for MD runner."
                # This writes predefined task configuration files such as ase.json.
                self.add_task(self._make_ase_task(cc, tt, pp))
        elif self.runner == "replica":
            for cc in confs:
                # This writes predefined task configuration files such as ase.json.
                self.add_task(self._make_ase_task(cc,))
        else:
            raise ValueError(f"Unknown md runner type: {self.runner}")
        return self

    def _make_ase_task(
        self, conf: str, temp: Optional[float] = None, press: Optional[float] = None,
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
        if self.runner == "md":
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
        elif self.runner == "replica":
            ase_input = REMDParameters(
                fmax=self.fmax,
                max_relax_steps=self.max_relax_steps,
                relax_is_slab=self.relax_is_slab,
                min_temperature=self.min_temperature,
                max_temperature=self.max_temperature,
                num_temperature_steps=self.num_temperature_steps,
                timestep_fs=self.timestep_fs,
                friction_fs=self.friction_fs,
                num_cycles=self.num_cycles,
                md_steps_per_cycle=self.md_steps_per_cycle,
                log_every_cycles=self.log_every_cycles,
                drop_start_fraction=self.drop_start_fraction,
                num_samples=self.num_samples,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown md runner type: {self.runner}")
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
        doc_runner = "MD runner type, 'md' or 'replica'."

        doc_temps = "List of temperatures for MD simulation."
        doc_press = "List of pressures for MD simulation (optional)."
        doc_ens = "Ensemble type (e.g., 'npt', 'nvt')."
        doc_dt = "MD time step (fs or ps)."
        doc_nsteps = "Number of MD steps."
        doc_trj_freq = "Trajectory output frequency."
        doc_tau_t = "Thermostat time constant."
        doc_tau_p = "Barostat time constant."
        doc_no_pbc = "Disable periodic boundary conditions."

        # Documents for ReplicaRunner parameters added similarly here.
        doc_fmax = "Maximum force for relaxation."
        doc_max_relax_steps = "Maximum relaxation steps."
        doc_relax_is_slab = "Indicate if the system is a slab."
        doc_min_temperature = "Minimum temperature for REMD."
        doc_max_temperature = "Maximum temperature for REMD."
        doc_num_temperature_steps = "Number of temperature steps for REMD."
        doc_timestep_fs = "Time step in femtoseconds for REMD."
        doc_friction_fs = "Friction coefficient in femtoseconds for REMD."
        doc_num_cycles = "Number of cycles for REMD."
        doc_md_steps_per_cycle = "MD steps per cycle for REMD."
        doc_log_every_cycles = "Logging frequency in cycles for REMD."
        doc_drop_start_fraction = "Fraction of initial data to drop in REMD."
        doc_num_samples = "Number of samples to collect in REMD."
        doc_seed = "Random seed for REMD."

        return [
            Argument("runner", str, optional=True, default="md", doc=doc_runner),
            Argument("temps", list, optional=True, default=None, doc=doc_temps),
            Argument("press", list, optional=True, default=None, doc=doc_press),
            Argument("ens", str, optional=True, default="npt", doc=doc_ens),
            Argument("dt", float, optional=True, default=2, doc=doc_dt),
            Argument("nsteps", int, optional=True, default=1000, doc=doc_nsteps),
            Argument("trj_freq", int, optional=True, default=100, doc=doc_trj_freq),
            Argument("tau_t", float, optional=True, default=100, doc=doc_tau_t),
            Argument("tau_p", float, optional=True, default=500, doc=doc_tau_p),
            Argument("no_pbc", bool, optional=True, default=False, doc=doc_no_pbc),
            Argument("fmax", float, optional=True, default=0.01, doc=doc_fmax),
            Argument("max_relax_steps", int, optional=True, default=2000, doc=doc_max_relax_steps),
            Argument("relax_is_slab", bool, optional=True, default=True, doc=doc_relax_is_slab),
            Argument("min_temperature", float, optional=True, default=300.0, doc=doc_min_temperature),
            Argument("max_temperature", float, optional=True, default=1000.0, doc=doc_max_temperature),
            Argument("num_temperature_steps", int, optional=True, default=5, doc=doc_num_temperature_steps),
            Argument("timestep_fs", float, optional=True, default=2.0, doc=doc_timestep_fs),
            Argument("friction_fs", float, optional=True, default=0.02, doc=doc_friction_fs),
            Argument("num_cycles", int, optional=True, default=200, doc=doc_num_cycles),
            Argument("md_steps_per_cycle", int, optional=True, default=250, doc=doc_md_steps_per_cycle),
            Argument("log_every_cycles", int, optional=True, default=10, doc=doc_log_every_cycles),
            Argument("drop_start_fraction", float, optional=True, default=0.2, doc=doc_drop_start_fraction),
            Argument("num_samples", int, optional=True, default=20, doc=doc_num_samples),
            Argument("seed", int, optional=True, default=None, doc=doc_seed),
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

        In configurations file (usually input.json), these are under each stage in exploration/stages.

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
