import unittest
from pymatgen.core import Structure, Lattice, Site, Species
from pfd.utils import structure_utils

class TestStructureUtils(unittest.TestCase):
    def setUp(self):
        # Simple cubic structure with 2 atoms close together
        lattice = Lattice.cubic(3.0)
        self.struct = Structure(
            lattice, ["Li", "O"], [[0, 0, 0], [0.1, 0.1, 0.1]]
        )
        # Large cell: two atoms close, one far away
        big_lattice = Lattice.cubic(100.0)
        self.struct_with_isolated = Structure(
            big_lattice, ["Li", "O"], [[0, 0, 0], [0.01, 0, 0]]
        )
        self.struct_with_isolated.append("H", [50, 50, 50], coords_are_cartesian=True)

    def test_remove_isolated_atoms(self):
        # Should remove the isolated atom
        struct_no_iso = structure_utils.remove_isolated_atoms(self.struct_with_isolated, radius=2.0)
        self.assertEqual(len(struct_no_iso), 2)
        # Should not remove any atom if all are close
        struct_no_iso2 = structure_utils.remove_isolated_atoms(self.struct, radius=2.0)
        self.assertEqual(len(struct_no_iso2), 2)

    def test_get_site_charge(self):
        # Elemental site
        site = self.struct[0]
        self.assertEqual(structure_utils.get_site_charge(site), 0)
        # Site with oxidation state
        site_oxi = Site(Species("Fe", 2), [0, 0, 0])
        self.assertEqual(structure_utils.get_site_charge(site_oxi), 2)

    def test_get_z_range_indices(self):
        # All atoms in [0,1) in z
        indices = structure_utils.get_z_range_indices(self.struct, 0.0, 1.0)
        self.assertEqual(set(indices), set([0, 1]))
        # No atoms in [0.5,0.6)
        indices = structure_utils.get_z_range_indices(self.struct, 0.5, 0.6)
        self.assertEqual(indices, [])
        # Atom at z=0.9 in [0.8,1.0)
        indices = structure_utils.get_z_range_indices(self.struct, 0.05, 0.11)
        self.assertEqual(indices, [1])

if __name__ == "__main__":
    unittest.main()
