import numpy as np
from ase.io import read,write
from ase import Atoms
from ase.build import make_supercell
from typing import List,Tuple,Union,Optional

def get_cell_perturb_matrix(cell_pert_fraction: float):
    """[Modified from dpdata]

    Args:
        cell_pert_fraction (float): The fraction of cell perturbation.

    Raises:
        RuntimeError: If cell_pert_fraction is negative.

    Returns:
        np.ndarray: A 3x3 cell perturbation matrix.
    """
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = np.random.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = np.array(
        [
            [1 + e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], 1 + e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], 1 + e[2]],
        ]
    )
    return cell_pert_matrix


def get_atom_perturb_vector(
    atom_pert_distance: float,
    atom_pert_style: str = "normal",
):
    """[Modified from dpdata] Perturb atom coord vectors.

    Args:
        atom_pert_distance (float): The distance to perturb the atom.
        atom_pert_style (str, optional): The style of perturbation. Defaults to "normal".

    Raises:
        RuntimeError: If atom_pert_distance is negative.
        RuntimeError: If atom_pert_style is not supported.

    Returns:
        np.ndarray: The perturbation vector for the atom.
    """
    random_vector = None
    if atom_pert_distance < 0:
        raise RuntimeError("atom_pert_distance can not be negative")

    if atom_pert_style == "normal":
        # return 3 numbers independently sampled from normal distribution
        e = np.random.randn(3)
        random_vector = (atom_pert_distance / np.sqrt(3)) * e
    elif atom_pert_style == "uniform":
        e = np.random.randn(3)
        while np.linalg.norm(e) < 0.1:
            e = np.random.randn(3)
        random_unit_vector = e / np.linalg.norm(e)
        v0 = np.random.rand(1)
        v = np.power(v0, 1 / 3)
        random_vector = atom_pert_distance * v * random_unit_vector
    elif atom_pert_style == "const":
        e = np.random.randn(3)
        while np.linalg.norm(e) < 0.1:
            e = np.random.randn(3)
        random_unit_vector = e / np.linalg.norm(e)
        random_vector = atom_pert_distance * random_unit_vector
    else:
        raise RuntimeError(f"unsupported options atom_pert_style={atom_pert_style}")
    return random_vector


def perturbed_atoms(
    atoms:Atoms,
    pert_num: int,
    cell_pert_fraction: float,
    atom_pert_distance: float,
    atom_pert_style: str = "normal",
    atom_pert_prob: float = 1.0,
):
    """[Modified from dpdata] Generate perturbed atoms.

    Args:
        atoms_ls (Union[List[Atoms],Atoms]): The input atoms to perturb.
    """
        
    pert_atoms_ls = []
    for ii in range(pert_num):
        cell_perturb_matrix = get_cell_perturb_matrix(cell_pert_fraction)
        frac_positions = atoms.get_scaled_positions()
        pert_cell = np.matmul(atoms.get_cell().array, cell_perturb_matrix)
        pert_positions = np.dot(frac_positions, pert_cell)
        pert_natoms = int(atom_pert_prob * len(atoms))
        pert_atom_id = sorted(
            np.random.choice(
                range(len(atoms)),
                pert_natoms,
                replace=False,
            ).tolist()
        )

        for kk in pert_atom_id:
            atom_perturb_vector = get_atom_perturb_vector(
                atom_pert_distance, atom_pert_style
                )
            pert_positions[kk] += atom_perturb_vector
            
        pert_atoms = Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=pert_positions,
            cell=pert_cell,
            pbc=atoms.get_pbc()
        )
        pert_atoms_ls.append(pert_atoms)
    return pert_atoms_ls


def perturb(
    atoms_ls: Union[List[Atoms],Atoms],
    pert_num: int,
    cell_pert_fraction: float,
    atom_pert_distance: float,
    atom_pert_style: str = "normal",
    atom_pert_prob: float = 1.0,
    supercell: Optional[Union[int, Tuple[int,int,int]]] = None,
) -> List[Atoms]:
    """[Modified from dpdata] Generate perturbed atoms.

    Args:
        atoms_ls (Union[List[Atoms],Atoms]): The input atoms to perturb.
        pert_num (int): The number of perturbed structures to generate for each input structure.
        cell_pert_fraction (float): The fraction of cell perturbation.
        atom_pert_distance (float): The distance to perturb the atom.
        atom_pert_style (str, optional): The style of perturbation. Defaults to "normal".
        atom_pert_prob (float, optional): The probability of perturbing each atom. Defaults to 1.0.

    Raises:
        RuntimeError: If atoms_ls is not a list or Atoms object.

    Returns:
        List[Atoms]: The list of perturbed atoms.
    """
    pert_atoms_ls = []
    if isinstance(atoms_ls, Atoms):
        atoms_ls = [atoms_ls]
    if supercell:
        if isinstance(supercell, int):
            supercell = (supercell, supercell, supercell)
        elif len(supercell) == 1:
            supercell = (supercell[0], supercell[0], supercell[0])
        else:
            supercell = tuple(supercell)
        atoms_ls = [atoms*supercell for atoms in atoms_ls]

    for atoms in atoms_ls:
        pert_atoms = perturbed_atoms(
            atoms,
            pert_num,
            cell_pert_fraction,
            atom_pert_distance,
            atom_pert_style,
            atom_pert_prob,
        )
        pert_atoms_ls.extend(pert_atoms)
    return pert_atoms_ls
