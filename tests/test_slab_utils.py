import os
import unittest
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import Slab
from pfd.utils import slab_utils


def check_isolated_site(
        structure: Structure,
        r: float=3.0,
) -> bool:
    """Check whether a structure contains isolated sites."""
    for site in structure.sites:
        nns = structure.get_neighbors(site, r)
        if len(nns) == 0:
            return True
    return False


class TestSlabUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(test_dir, "data")
        cls.si = Structure.from_file(os.path.join(data_dir, "Si.cif"))
        cls.tio2 = Structure.from_file(os.path.join(data_dir, "TiO2.cif"))

    def test_get_slabs_si(self):
        # Test (100), (110), (111) for Si
        for miller in [(1,0,0), (1,1,0), (1,1,1)]:
            slabs_si = slab_utils.get_slabs(self.si, miller, min_slab=10, min_vac=8)
            self.assertTrue(len(slabs_si) > 0)
            for slab in slabs_si:
                self.assertIsInstance(slab, Slab)
                self.assertGreater(len(slab), 0)
                self.assertGreaterEqual(slab.lattice.c, 18)
                self.assertGreaterEqual(slab.lattice.a, 12)
                self.assertGreaterEqual(slab.lattice.b, 12)

        # Test (100), (110), (001) for TiO2
        for miller in [(1,0,0), (1,1,0), (0,0,1)]:
            slabs = slab_utils.get_slabs(self.tio2, miller, min_slab=10, min_vac=10)
            self.assertTrue(len(slabs) > 0)
            for slab in slabs:
                self.assertIsInstance(slab, Slab)
                self.assertGreater(len(slab), 0)
                self.assertGreaterEqual(slab.lattice.c, 18)
                self.assertGreaterEqual(slab.lattice.a, 12)
                self.assertGreaterEqual(slab.lattice.b, 12)
                self.assertFalse(slab.is_polar())  # Should have corrected and dropped all polar surfaces.

    def test_surface_mapping_and_symmetry(self):
        # Use Si (111) slab, check surface mapping and symmetry removal
        slabs = slab_utils.get_slabs(self.si, (1,1,1), min_slab=6, min_vac=8)
        slab = slabs[0]  # This should have symmetry.
        surf = slab.get_surface_sites()
        # 111 surfaces are related by inverse, not mirror.
        mapping = slab_utils.get_slab_equivalence_groups(slab.species, surf, mode="mirror")
        self.assertTrue(mapping is None)
        mapping = slab_utils.get_slab_equivalence_groups(slab.species, surf, mode="inverse")
        self.assertTrue(mapping is not None)
        self.assertEqual(len(mapping), len(surf["top"]))
        # Remove one top site and its symmetric bottom
        top_inds = list(mapping.keys())
        new_slab = slab_utils.symmetrically_remove_top_sites(slab, [top_inds[0]], mapping)
        self.assertEqual(len(new_slab), len(slab)-2)

    def test_remove_random_surface_sites(self):
        # Use TiO2 (110) slab, remove 50% of surface sites
        slabs = slab_utils.get_slabs(self.tio2, (1,1,0), min_slab=6, min_vac=8)
        slab = slabs[0]
        surf = slab.get_surface_sites()
        names, vac_slabs, no_sym = slab_utils.remove_random_surface_sites(
            slab, surf, 0.5, n_sample_max=2
        )
        self.assertEqual(len(names), len(vac_slabs))
        self.assertEqual(len(names), 2)
        # Should have symmetry.
        self.assertFalse(no_sym)
        self.assertEqual(len(names), len(vac_slabs))
        for vac in vac_slabs:
            self.assertLess(len(vac), len(slab))
            self.assertGreater(len(vac), 0)
            # Check no isolated site.
            self.assertFalse(check_isolated_site(vac))

        # Perform removal on only oxygen atoms.
        names, vac_slabs, no_sym = slab_utils.remove_random_surface_sites(
            slab, surf, 0.5, n_sample_max=2, remove_atom_types=["O2-"],
        )
        # Should have symmetry.
        self.assertFalse(no_sym)
        self.assertEqual(len(names), len(vac_slabs))
        for vac in vac_slabs:
            self.assertLess(len(vac), len(slab))
            self.assertGreater(len(vac), 0)
            # Check no isolated site.
            self.assertFalse(check_isolated_site(vac))
            # Check only O2- has been removed.
            self.assertEqual(vac.composition["Ti4+"], slab.composition["Ti4+"])
            self.assertLess(vac.composition["O2-"], slab.composition["O2-"])

        # If not remove isolated atom, real_ratio should be the same as expected ratio.
        names, vac_slabs, no_sym = slab_utils.remove_random_surface_sites(
            slab, surf, 0.3, n_sample_max=2, remove_isolated_atom=False
        )
        # name example: remove_0.5000_sample_0_actual_0.5123
        for name in names:
            expected_ratio = float(name.split("_")[1])
            actual_ratio = float(name.split("_")[-1])
            self.assertAlmostEqual(expected_ratio, actual_ratio, places=4)


    def test_generate_slabs_with_random_vacancies(self):
        # Use Si (100) slab, generate slabs with 0 and 0.5 vacancy ratio
        names, vac_slabs, slabs = slab_utils.generate_slabs_with_random_vacancies(
            self.si, (1,0,0),
            min_slab=12, min_vac=20,
            min_vacancy_ratio=0,
            max_vacancy_ratio=0.5,
            num_vacancy_ratios=3,
            n_sample_per_ratio=1
        )
        self.assertTrue(len(names) >= 2)
        self.assertTrue(all(isinstance(s, Slab) for s in vac_slabs))
        self.assertEqual(len(names), len(vac_slabs))
        self.assertEqual(len(names), 3 * len(slabs))  # 3 vacancy ratios per slab, 1 sample each.
        self.assertTrue(all("miller_1_0_0" in name for name in names))

        # Check whether log catches warning because this is ionic removal.
        with self.assertLogs(level='WARNING') as cm:
            names, vac_slabs, slabs = slab_utils.generate_slabs_with_random_vacancies(
                self.tio2, (1, 0, 0),
                min_slab=12, min_vac=20,
                min_vacancy_ratio=0,
                max_vacancy_ratio=0.5,
                num_vacancy_ratios=3,
                n_sample_per_ratio=1
            )
        self.assertTrue(any('warn' in m.lower() for m in cm.output))
        # In this particular case, should find 2 slabs (due to polarity correction).
        self.assertEqual(len(slabs), 2)


if __name__ == "__main__":
    unittest.main()
