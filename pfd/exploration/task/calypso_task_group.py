import copy
import logging
import random
from typing import (
    List,
    Dict,
)

import numpy as np
from dargs import Argument

from pfd.constants import (
    calypso_input_file,
    ase_input_name,
    calypso_check_opt_file
)
from pfd.exploration.md.ase import MDParameters

from .calypso import (
    make_calypso_input,
)
from .task import (
    ExplorationTask,
)
from .task_group import (
    ExplorationTaskGroup,
)

atomic_symbols = (
    'X',  # placeholder
    'H', 'He',
    'Li', 'Be', 'B',  'C',  'N', 'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
)  # fmt: skip
atomic_number_map = {key: value for value, key in enumerate(atomic_symbols)}
# Covalent radii from:
#
#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
UNKN = 0.2
covalent_radii = [
    # X, placeholder
    UNKN,
    # H    He
    0.31, 0.28,
    # Li    Be     B     C     N     O     F    Ne
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    # Na    Mg    Al    Si     P     S    Cl    Ar
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    #  K    Ca    Sc    Ti     V    Cr    Mn    Fe    Co    Ni    Cu    Zn    Ga    Ge    As    Se    Br    Kr
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    # Rb    Sr     Y    Zr    Nb    Mo    Tc    Ru    Rh    Pd    Au    Cd    In    Sn    Sb    Te     I    Xe
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
    # Cs    Ba
    2.44, 2.15,
    # La    Ce    Pr    Nd    Pm    Sm    Eu    Gd    Tb    Dy    Ho    Er    Tm    Yb    Lu
    2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,
    # Hf    Ta     W    Re    Os    Ir    Pt    Au    Hg    Tl    Pb    Bi    Po    At    Rn
    1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    # Fr    Ra
    2.60, 2.21,
    # Ac    Th    Pa     U    Np    Pu    Am    Cm    Bk    Cf    Es    Fm    Md    No    Lr
    2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN,
    # Rf    Db    Sg    Bh    Hs    Mt    Ds    Rg    Cn    Nh    Fl    Mc    Lv    Ts    Og
    UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN,
]  # fmt: skip


class CalyTaskGroup(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()

    def set_params(
        self,
        numb_of_species,
        name_of_atoms,
        numb_of_atoms,
        distance_of_ions=None,
        atomic_number=None,
        pop_size: int = 30,
        max_step: int = 5,
        system_name: str = "CALYPSO",
        numb_of_formula: List[int] = [1, 1],
        pressure: float = 0.001,
        fmax: float = 0.01,
        volume: float = 0,
        ialgo: int = 2,
        pso_ratio: float = 0.6,
        icode: int = 15,
        numb_of_lbest: int = 4,
        numb_of_local_optim: int = 4,
        command: str = "sh submit.sh",
        max_time: int = 9000,
        gen_type: int = 1,
        pick_up: bool = False,
        pick_step: int = 1,
        parallel: bool = False,
        split: bool = True,
        spec_space_group: List[int] = [2, 230],
        vsc: bool = False,
        ctrl_range: List[List[int]] = [[1, 10]],
        max_numb_atoms: int = 100,
        opt_step: int = 1000,
        ens: str = "lbfgs",
        **kwargs
        
    ):
        """
        Set calypso parameters
        """
        self.numb_of_species = numb_of_species
        self.numb_of_atoms = numb_of_atoms

        if isinstance(name_of_atoms, list) and all(
            [isinstance(i, list) for i in name_of_atoms]
        ):
            overlap = set(name_of_atoms[0])
            for temp in name_of_atoms[1:]:
                overlap = overlap & set(temp)

            if any(map(lambda s: (set(s) - overlap) == 0, name_of_atoms)):
                raise ValueError(
                    f"Any sub-list should not equal with intersection, e.g. [[A,B,C], [B,C], [C]] is not allowed."
                )

            while True:
                choice = []
                for _atoms in name_of_atoms:
                    value = random.choice(_atoms)
                    logging.info(
                        f"randomly choose {value} from {_atoms}, already choose: {choice}"
                    )
                    if value in choice:
                        break
                    choice.append(value)
                else:
                    break
            self.name_of_atoms = choice
            logging.info(f"The final choice is {self.name_of_atoms}")
            self.atomic_number = [atomic_symbols.index(i) for i in self.name_of_atoms]
        else:
            self.name_of_atoms = name_of_atoms
            self.atomic_number = atomic_number

        if isinstance(distance_of_ions, dict):
            updated_table = copy.deepcopy(covalent_radii)
            for key, value in distance_of_ions.items():
                updated_table[atomic_number_map[key]] = value

            temp_distance_mtx = np.zeros((numb_of_species, numb_of_species))
            for i in range(numb_of_species):
                for j in range(numb_of_species):
                    temp_distance_mtx[i][j] = round(
                        updated_table[atomic_number_map[self.name_of_atoms[i]]] * 0.7
                        + updated_table[atomic_number_map[self.name_of_atoms[j]]] * 0.7,
                        2,
                    )
            self.distance_of_ions = temp_distance_mtx
        else:
            self.distance_of_ions = distance_of_ions

        self.pop_size = pop_size
        self.max_step = max_step
        self.system_name = system_name
        self.numb_of_formula = numb_of_formula
        self.pressure = pressure
        self.fmax = fmax
        self.volume = volume
        self.ialgo = ialgo
        self.pso_ratio = pso_ratio
        self.icode = icode
        self.numb_of_lbest = numb_of_lbest
        self.numb_of_local_optim = numb_of_local_optim
        self.command = command
        self.max_time = max_time
        self.gen_type = gen_type
        self.pick_up = pick_up
        self.pick_step = pick_step
        self.parallel = parallel
        self.split = split
        self.spec_space_group = spec_space_group
        self.vsc = vsc
        self.ctrl_range = ctrl_range
        self.max_numb_atoms = max_numb_atoms
        self.opt_step = opt_step
        self.caly_set = True
        self.ens = ens

    def make_task(self) -> ExplorationTaskGroup:
        """
        Make the CALYPSO task group.

        Returns
        -------
        task_grp: ExplorationTaskGroup
            Return one calypso task group.
        """
        if not self.caly_set:
            raise RuntimeError("calypso settings are not set")
        # clear all existing tasks
        self.clear()
        self.add_task(self._make_caly_task())
        return self

    def _make_caly_task(self) -> ExplorationTask:
        input_file_str, check_opt_str = make_calypso_input(
            self.numb_of_species,
            self.name_of_atoms,
            self.atomic_number,
            self.numb_of_atoms,
            self.distance_of_ions,
            self.pop_size,
            self.max_step,
            self.system_name,
            self.numb_of_formula,
            self.volume,
            self.ialgo,
            self.pso_ratio,
            self.icode,
            self.numb_of_lbest,
            self.numb_of_local_optim,
            self.command,
            self.max_time,
            self.gen_type,
            self.pick_up,
            self.pick_step,
            self.parallel,
            self.split,
            self.spec_space_group,
            self.vsc,
            self.ctrl_range,
            self.max_numb_atoms,
            opt_step=self.opt_step,
        )
        task = ExplorationTask()
        task.add_file(calypso_input_file, input_file_str)
        # add json files
        opt_input = MDParameters(
            scalar_pressure=self.pressure,
            max_step=self.opt_step,
            ensemble=self.ens,
            fmax=self.fmax,
        )
        task.add_file(ase_input_name, opt_input.to_json())
        task.add_file(calypso_check_opt_file, check_opt_str)
        return task
    
    @classmethod
    def normalize_config(cls, data: Dict = {}, strict: bool = False) -> Dict:
        r"""Normalize the argument.

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
        r"""The argument definition for CALYPSO task group parameters.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments for CALYPSO task setup.
        """

        doc_numb_of_species = "Number of different atomic species in the system."
        doc_name_of_atoms = "List of atomic symbols (e.g., ['Li', 'O']) or nested lists for random selection."
        doc_numb_of_atoms = "List of number of atoms for each species."
        doc_distance_of_ions = "Distance matrix between ions or dict of covalent radii adjustments. If None, uses default covalent radii."
        doc_atomic_number = "List of atomic numbers for each species. Auto-generated if None."
        doc_pop_size = "Population size for CALYPSO genetic algorithm."
        doc_max_step = "Maximum number of CALYPSO evolution steps."
        doc_system_name = "Name of the system for CALYPSO."
        doc_numb_of_formula = "Number of formula units range [min, max]."
        doc_pressure = "External pressure for ASE optimization (GPa)."
        doc_fmax = "Maximum force criterion for structure optimization (eV/Å)."
        doc_volume = "Volume constraint for structure generation. 0 means no constraint."
        doc_ialgo = "CALYPSO algorithm type (1 or 2)."
        doc_pso_ratio = "PSO (Particle Swarm Optimization) ratio."
        doc_icode = "Local optimization code (e.g., 1=VASP, 2=SIESTA, 15=ASE)."
        doc_numb_of_lbest = "Number of local best structures to keep."
        doc_numb_of_local_optim = "Number of structures to perform local optimization."
        doc_command = "Command to execute for structure optimization."
        doc_max_time = "Maximum time allowed for each optimization (seconds)."
        doc_gen_type = "Structure generation type."
        doc_pick_up = "Whether to pick up from previous run."
        doc_pick_step = "Step number to pick up from."
        doc_parallel = "Enable parallel CALYPSO execution."
        doc_split = "Split CALYPSO tasks."
        doc_spec_space_group = "Specified space group range [min, max]."
        doc_vsc = "Variable stoichiometry composition."
        doc_ctrl_range = "Control range for variable composition [[min, max]]."
        doc_max_numb_atoms = "Maximum number of atoms in generated structures."
        doc_opt_step = "Maximum optimization steps for ASE."
        doc_ens = "Optimizer/ensemble for ASE (e.g., 'lbfgs', 'bfgs', 'fire')."

        return [
            Argument("numb_of_species", int, optional=False, doc=doc_numb_of_species),
            Argument("name_of_atoms", [list, str], optional=False, doc=doc_name_of_atoms),
            Argument("numb_of_atoms", list, optional=False, doc=doc_numb_of_atoms),
            Argument("distance_of_ions", [dict, list, type(None)], optional=True, default=None, doc=doc_distance_of_ions),
            Argument("atomic_number", [list, type(None)], optional=True, default=None, doc=doc_atomic_number),
            Argument("pop_size", int, optional=True, default=30, doc=doc_pop_size),
            Argument("max_step", int, optional=True, default=5, doc=doc_max_step),
            Argument("system_name", str, optional=True, default="CALYPSO", doc=doc_system_name),
            Argument("numb_of_formula", list, optional=True, default=[1, 1], doc=doc_numb_of_formula),
            Argument("pressure", float, optional=True, default=0.001, doc=doc_pressure),
            Argument("fmax", float, optional=True, default=0.01, doc=doc_fmax),
            Argument("volume", float, optional=True, default=0, doc=doc_volume),
            Argument("ialgo", int, optional=True, default=2, doc=doc_ialgo),
            Argument("pso_ratio", float, optional=True, default=0.6, doc=doc_pso_ratio),
            Argument("icode", int, optional=True, default=15, doc=doc_icode),
            Argument("numb_of_lbest", int, optional=True, default=4, doc=doc_numb_of_lbest),
            Argument("numb_of_local_optim", int, optional=True, default=4, doc=doc_numb_of_local_optim),
            Argument("command", str, optional=True, default="sh submit.sh", doc=doc_command),
            Argument("max_time", int, optional=True, default=9000, doc=doc_max_time),
            Argument("gen_type", int, optional=True, default=1, doc=doc_gen_type),
            Argument("pick_up", bool, optional=True, default=False, doc=doc_pick_up),
            Argument("pick_step", int, optional=True, default=1, doc=doc_pick_step),
            Argument("parallel", bool, optional=True, default=False, doc=doc_parallel),
            Argument("split", bool, optional=True, default=True, doc=doc_split),
            Argument("spec_space_group", list, optional=True, default=[2, 230], doc=doc_spec_space_group),
            Argument("vsc", bool, optional=True, default=False, doc=doc_vsc),
            Argument("ctrl_range", list, optional=True, default=[[1, 10]], doc=doc_ctrl_range),
            Argument("max_numb_atoms", int, optional=True, default=100, doc=doc_max_numb_atoms),
            Argument("opt_step", int, optional=True, default=1000, doc=doc_opt_step),
            Argument("ens", str, optional=True, default="lbfgs", doc=doc_ens),
        ]
    
    @classmethod
    def make_task_grp(cls, **kwargs) -> "CalyTaskGroup":
        task_grp = cls()
        normalized_config = cls.normalize_config(kwargs, strict=False)
        task_grp.set_params(**normalized_config)
        task_grp.make_task()
        return task_grp
    
    @classmethod
    def make_task_grp_from_conf(
        cls,
        task_grp_config: Dict,
        *args,
        **kwargs
    ) -> "CalyTaskGroup":
        """
        Create Calypso task group from configuration files and task group config.
        
        Parameters
        ----------
        init_confs : List[str]
            List of paths to initial configuration files (not used for Calypso)
        task_grp_config : Dict
            Task group configuration containing all Calypso parameters
            
        Returns
        -------
        CalyTaskGroup
            Configured Calypso task group
        """
        # Remove conf_idx and n_sample as they're not used in Calypso
        task_grp_config.pop("conf_idx", None)
        task_grp_config.pop("n_sample", None)
        
        return cls.make_task_grp(**task_grp_config)