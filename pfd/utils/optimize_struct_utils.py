"""Optimize structures utility functions."""
from typing import Optional

from ase import Atoms
from ase.calculators.calculator import Calculator

from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGSLineSearch

from pfd.constants import ase_log_name


def relax_structure_ase(
        atoms_init: Atoms,
        calculator: Calculator,
        fmax_final: float = 0.01,
        max_steps: int = 2000,
        is_slab: bool = False,
        clear_calculator: bool = False,
        logfile: str = ase_log_name,
) -> Atoms:
    """Relax structure with given calculator in ASE.

    Args:
        atoms_init (Atoms): Initial structure to be relaxed.
        calculator (Calculator): ASE calculator to use for relaxation.
        fmax_final (float, optional): Final maximum force convergence criterion. Defaults to 0.01.
        max_steps (int, optional): Maximum number of optimization steps. Defaults to 2000.
        is_slab (bool, optional): Whether the structure is a slab/interface (i.e. contains vacuum layer).
            Defaults to False.
            When dealing with pymatgen generated grain boundaries, set this to True.
        clear_calculator (bool, optional):
            Whether to clear attached claculator after relaxation. Defaults to False.
        logfile (str, optional): Log file name for optimization. Defaults to pfd.constants.ase_log_name.
    Returns:
        Atoms: Relaxed structure.
    """

    atoms = atoms_init.copy()
    atoms.calc = calculator
    # For slab, must conserve volume to prevent eliminating vacuum.
    filt = FrechetCellFilter(atoms, constant_volume=is_slab)
    # Coarse optimization with FIRE.
    opt1 = FIRE(filt, logfile=logfile)
    opt1.run(fmax=fmax_final * 3, steps=max_steps//2)
    opt2 = LBFGSLineSearch(filt, logfile=logfile)
    converged = opt2.run(fmax=fmax_final, steps=max_steps)
    if not converged:
        raise RuntimeError(
            f"Structure optimization failed to converge in {max_steps} steps."
            " Check validity of initial structure!"
        )
    if clear_calculator:
        atoms.calc = None  # Clear calculator.
    return atoms
