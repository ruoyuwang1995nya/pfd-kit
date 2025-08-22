import ase
from ase import Atoms
import numpy as np
import re
from typing import List, Dict, Union, Optional,Tuple
import random
class LabeledAtoms(Atoms):
    """A subclass of ASE Atoms that includes labels for each atom."""
    
    def __init__(self, 
                 *args, 
                 forces: Optional[np.ndarray] = None,
                 energy: Optional[float] = None,
                 virials: Optional[np.ndarray] = None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        
        self._energy = energy
        
        self._forces = forces
        if self._forces is not None:
            assert self._forces.shape == (len(self), 3), "Forces must be of shape (num_atoms, 3)"

        self._virials = virials
        if self._virials is not None:
            assert self._virials.shape == (len(self),3, 3), "Virials must be of shape (3, 3)"

    @property
    def forces(self):
        """Return the forces acting on the atoms."""
        return self._forces

    @property
    def energy(self):
        """Return the energies of the atoms."""
        return self._energy

    @property
    def virials(self):
        """Return the virials of the atoms."""
        return self._virials
    
    @classmethod
    def from_atoms(
        cls, 
        atoms: Atoms, 
        energy: Optional[float] = None,
        forces: Optional[np.ndarray] = None,
        virials: Optional[np.ndarray] = None,
        **kwargs) -> 'LabeledAtoms':
        """
        Create a LabeledAtoms object from an existing ASE Atoms object.
    
        Parameters
        ----------
        atoms : Atoms
            The ASE Atoms object to convert
        energy : float, optional
            Total energy of the system. If None, will try to get from atoms.info['energy'] 
            or atoms.calc.results['energy']
        forces : np.ndarray, optional
            Forces on atoms with shape (num_atoms, 3). If None, will try to get from 
            atoms.arrays['forces'] or atoms.calc.results['forces']
        virials : np.ndarray, optional
            Virial tensor with shape (num_atoms, 3, 3) or (3, 3). If None, will try to get 
            from atoms.info['stress'] or atoms.calc.results['stress']
        **kwargs
            Additional keyword arguments passed to LabeledAtoms constructor
        
        Returns
        -------
        LabeledAtoms
            New LabeledAtoms object with the specified properties
        
        Examples
        --------
        >>> atoms = read('structure.cif')
        >>> labeled = LabeledAtoms.from_atoms(atoms, energy=-123.45)
    
        >>> # From atoms with calculator results
        >>> atoms.calc = some_calculator
        >>> energy = atoms.get_potential_energy()
        >>> forces = atoms.get_forces()
        >>> labeled = LabeledAtoms.from_atoms(atoms)  # Auto-extracts properties
        """
        # Extract basic structural information
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
    
    # Try to extract energy from various sources
        if energy is None:
            # Priority: explicit parameter > info dict > calculator results
            if 'energy' in atoms.info:
                energy = atoms.info['energy']
            elif hasattr(atoms, 'calc') and atoms.calc is not None:
                if hasattr(atoms.calc, 'results') and 'energy' in atoms.calc.results:
                    energy = atoms.calc.results['energy']
    
    # Try to extract forces from various sources
        if forces is None:
            if 'forces' in atoms.arrays:
                forces = atoms.arrays['forces']
            elif hasattr(atoms, 'calc') and atoms.calc is not None:
                if hasattr(atoms.calc, 'results') and 'forces' in atoms.calc.results:
                    forces = atoms.calc.results['forces']
    
    # Try to extract virials/stress from various sources
        if virials is None:
        # Check for stress in info or calculator results
            stress = None
            if 'stress' in atoms.info:
                stress = atoms.info['stress']
            elif hasattr(atoms, 'calc') and atoms.calc is not None:
                if hasattr(atoms.calc, 'results') and 'stress' in atoms.calc.results:
                    stress = atoms.calc.results['stress']
    
    
    # Create LabeledAtoms object
        labeled_atoms = cls(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=pbc,
            energy=energy,
            forces=forces,
            virials=virials,
            **kwargs
        )
    
        return labeled_atoms
    
    
    
def read_labeled_xyz(filename: str) -> List[LabeledAtoms]:
    """Read an extended XYZ file and return a list of LabeledAtoms objects."""
    labeled_atoms_list = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            header = f.readline().strip()
            
            # Parse header properties
            props = {}
            # Split by spaces but keep quoted strings together
            header_parts = re.findall(r'(\w+="[^"]*"|\w+=\S+)', header)
            
            for item in header_parts:
                if '=' in item:
                    key, value = item.split('=', 1)
                    # Remove quotes if present
                    props[key] = value.strip('"')
            
            # Parse Properties field
            prop_list = []
            if 'Properties' in props:
                fields = props['Properties'].split(':')
                i = 0
                while i < len(fields):
                    name = fields[i]
                    typ = fields[i+1] 
                    size = int(fields[i+2])
                    prop_list.append((name, typ, size))
                    i += 3
            
            # Read atom lines
            atoms_data = []
            for _ in range(num_atoms):
                atom_line = f.readline().strip().split()
                atom_props = {}
                idx = 0
                for name, typ, size in prop_list:
                    vals = atom_line[idx:idx+size]
                    if typ == 'R':
                        vals = [float(v) for v in vals]
                    elif typ == 'I':
                        vals = [int(v) for v in vals]
                    # For single-value properties, store as scalar
                    atom_props[name] = vals[0] if size == 1 else vals
                    idx += size
                atoms_data.append(atom_props)
            
            # Extract data for LabeledAtoms creation
            symbols = [atom['species'] for atom in atoms_data]
            positions = np.array([atom['pos'] for atom in atoms_data])
            
            # Extract forces if present
            forces = None
            if 'forces' in atoms_data[0]:
                forces = np.array([atom['forces'] for atom in atoms_data])
            
            # Extract energy from header
            energy = None
            if 'energy' in props:
                energy = float(props['energy'])
            
            # Parse cell from Lattice string
            cell = None
            pbc = [True, True, True]  # Default
            
            if 'Lattice' in props:
                lattice_str = props['Lattice']
                # Convert string to floats and reshape to 3x3 matrix
                lattice_vals = [float(x) for x in lattice_str.split()]
                if len(lattice_vals) == 9:
                    # Lattice is given as 9 values: a1x a1y a1z a2x a2y a2z a3x a3y a3z
                    # Reshape to 3x3 matrix where each row is a lattice vector
                    cell = np.array(lattice_vals).reshape(3, 3)
                else:
                    raise ValueError(f"Expected 9 lattice parameters, got {len(lattice_vals)}")
            
            # Parse PBC from pbc field
            if 'pbc' in props:
                pbc_str = props['pbc']
                pbc = [x.upper() == 'T' for x in pbc_str.split()]
            
            # Create LabeledAtoms object
            labeled_atoms = LabeledAtoms(
                symbols=symbols,
                positions=positions,
                cell=cell,
                pbc=pbc,
                forces=forces,
                energy=energy
            )
            
            # Store any additional properties in info dict
            for key, value in props.items():
                if key not in ['energy', 'Lattice', 'Properties', 'pbc']:
                    try:
                        # Try to convert to float if it's a number
                        labeled_atoms.info[key] = float(value)
                    except ValueError:
                        # Keep as string if it's not a number
                        labeled_atoms.info[key] = value

            labeled_atoms_list.append(labeled_atoms)

    return labeled_atoms_list


def write_labeled_xyz(atoms: LabeledAtoms, filename: str, append: bool = True):
    """
    Write a LabeledAtoms object to extended XYZ format.
    
    Parameters
    ----------
    atoms : LabeledAtoms
        The atoms object to write
    filename : str
        Output filename
    append : bool, default=True
        If True, append to existing file; if False, overwrite
    """
    import re
    
    mode = 'a' if append else 'w'
    
    with open(filename, mode) as f:
        # Write number of atoms
        f.write(f"{len(atoms)}\n")
        
        # Build header line
        header_parts = []
        
        # Add lattice if present
        if atoms.cell is not None:
            cell_flat = atoms.cell.flatten()
            lattice_str = ' '.join(f"{x:.6f}" for x in cell_flat)
            header_parts.append(f'Lattice="{lattice_str}"')
        
        # Build Properties field
        properties = ['species:S:1', 'pos:R:3']
        if atoms.forces is not None:
            properties.append('forces:R:3')
        
        header_parts.append(f'Properties={":".join(properties)}')
        
        # Add energy if present
        if atoms.energy is not None:
            header_parts.append(f'energy={atoms.energy:.8f}')
        
        # Add PBC
        if hasattr(atoms, 'pbc') and atoms.pbc is not None:
            pbc_str = ' '.join('T' if p else 'F' for p in atoms.pbc)
            header_parts.append(f'pbc="{pbc_str}"')
        
        # Add any additional properties from info dict
        for key, value in atoms.info.items():
            if key not in ['energy']:  # energy already handled above
                if isinstance(value, (int, float)):
                    header_parts.append(f'{key}={value}')
                else:
                    header_parts.append(f'{key}="{value}"')
        
        # Write header line
        f.write(' '.join(header_parts) + '\n')
        
        # Write atom lines
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        forces = atoms.forces if atoms.forces is not None else None
        
        for i in range(len(atoms)):
            line_parts = [symbols[i]]
            
            # Add position
            line_parts.extend(f"{pos:.6f}" for pos in positions[i])
            
            # Add forces if present
            if forces is not None:
                line_parts.extend(f"{force:.6f}" for force in forces[i])
            
            f.write(' '.join(line_parts) + '\n')

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


def write_xyz_trajectory(atoms_list: List[LabeledAtoms], filename: str):
    """
    Write multiple LabeledAtoms objects to a single extended XYZ trajectory file.
    
    Parameters
    ----------
    atoms_list : List[LabeledAtoms]
        List of atoms objects to write
    filename : str
        Output filename
    """
    # Write first frame (overwrite)
    if atoms_list:
        write_labeled_xyz(atoms_list[0], filename, append=False)

        # Write remaining frames (append)
        for atoms in atoms_list[1:]:
            write_labeled_xyz(atoms, filename, append=True)