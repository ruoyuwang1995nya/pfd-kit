from ase import Atoms
from ase.data import chemical_symbols
import dpdata
import numpy as np
from typing import List, Dict, Union, Optional,Tuple
import random

def train_test_split(atoms_list: List[Atoms], 
                     test_size: float = 0.2, 
                     random_state: Optional[int] = None) -> Tuple[List[Atoms], List[Atoms]]:
    """
    Randomly split atoms_list into training and test sets.
    
    Parameters
    ----------
    atoms_list : List[LabeledAtoms]
        List of atoms objects
    test_size : float, default=0.2
        Fraction of data to use for testing (0.0 to 1.0)
    random_state : int, optional
        Random seed for reproducible splits
        
    Returns
    -------
    Tuple[List, List]
        (train_atoms_list, test_atoms_list)
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Simple random shuffle approach
    indices = list(range(len(atoms_list)))
    random.shuffle(indices)
    
    if test_size < 1:
        n_test = int(len(atoms_list) * test_size)
    else:
        n_test = int(test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_atoms = [atoms_list[i] for i in train_indices]
    test_atoms = [atoms_list[i] for i in test_indices]
    
    return train_atoms, test_atoms

def sort_atoms_by_atomic_number(atoms):
    """Sort atoms in place by atomic number."""
    atomic_numbers = atoms.get_atomic_numbers()
    # Get indices that would sort the atomic numbers
    sort_indices = np.argsort(atomic_numbers)
    # Create new atoms object with sorted order
    sorted_atoms = atoms[sort_indices]
    return sorted_atoms


def get_element_types_from_sorted_atoms(sorted_atoms):
    """Get unique element types from sorted atoms (one-liner approach)."""
    atomic_numbers = sorted_atoms.get_atomic_numbers()
    # dict.fromkeys preserves order while removing duplicates
    unique_z = list(dict.fromkeys(atomic_numbers))
    return [chemical_symbols[z] for z in unique_z]

def dpdata2ase(
    sys: dpdata.System
    )->List[Atoms]:
    """Convert dpdata System to ase.Atoms."""
    atoms_list = []
    for ii in range(len(sys)):
        atoms=Atoms(
            symbols=[sys.get_atom_names()[i] for i in sys.get_atom_types()],
            positions=sys[ii].data["coords"][0].tolist(),
            cell=sys[ii].data["cells"][0].tolist(),
            pbc= not sys[ii].nopbc
        )
        # set the virials and forces
        if "virial" in sys[ii].data:
            atoms.set_array("virial", sys[ii].data["virial"][0])
        if "forces" in sys[ii].data:
            atoms.set_array("force", sys[ii].data["forces"][0])
        if "energies" in sys[ii].data:
            atoms.info["energy"] = sys[ii].data["energies"][0]
        atoms_list.append(atoms)
    return atoms_list