"""Test optimization of ASE structure."""

import unittest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pfd.utils.optimize_struct_utils import relax_structure_ase


class TestRelaxStructureASE(unittest.TestCase):
    def setUp(self):
        # Initialize calculator and structure.
        self.atoms_init = bulk("Au", "fcc", a=5.0)
        self.calculator = EMT()

    def test_relax_structure_ase(self):
        # Relax the structure.
        atoms_relaxed = relax_structure_ase(
            self.atoms_init,
            self.calculator,
            fmax_final=0.01,
            max_steps=100,
            is_slab=False,
            clear_calculator=False,
            logfile="test_relax.log",
        )

        # Check whether lattice parameters changed.
        cell_init = self.atoms_init.get_cell().lengths()
        cell_relaxed = atoms_relaxed.get_cell().lengths()
        self.assertFalse(
            np.allclose(cell_init, cell_relaxed),
            "Lattice parameters did not change after relaxation."
        )
        # Check whether all of a, b and c changed.
        changes = np.abs(cell_relaxed - cell_init) > 1e-5
        self.assertTrue(
            np.all(changes),
            "Not all lattice parameters changed after relaxation."
        )

        # Check whether energy decreased.
        self.atoms_init.calc = self.calculator
        energy_init = self.atoms_init.get_potential_energy()
        energy_relaxed = atoms_relaxed.get_potential_energy()
        self.assertLess(
            energy_relaxed, energy_init,
            "Relaxed structure has higher energy than initial structure."
        )

        # Check maximum force is below threshold.
        forces = atoms_relaxed.get_forces()
        max_force = np.max(np.abs(forces))
        self.assertLess(
            max_force, 0.01,
            f"Max force {max_force} exceeds threshold after relaxation."
        )

        # Check log file is created.
        try:
            with open("test_relax.log", "r") as f:
                log_content = f.read()
            self.assertTrue(len(log_content) > 0, "Log file is empty.")
        except FileNotFoundError:
            self.fail("Log file was not created.")

    def test_relax_is_slab(self):
        # Create a slab structure.
        slab = self.atoms_init.repeat((2, 2, 3))
        slab.center(vacuum=10.0, axis=2)  # Add vacuum layer.

        # Relax the slab structure.
        atoms_relaxed = relax_structure_ase(
            slab,
            self.calculator,
            fmax_final=0.01,
            max_steps=1000,
            is_slab=True,
            clear_calculator=False,
            logfile="test_relax_slab.log",
        )

        # Check maximum force is below threshold.
        forces = atoms_relaxed.get_forces()
        max_force = np.max(np.abs(forces))
        self.assertLess(
            max_force, 0.01,
            f"Max force {max_force} exceeds threshold after slab relaxation."
        )

        # Check cell volume is conserved (within tolerance).
        cell_initial = slab.get_cell().volume
        cell_relaxed = atoms_relaxed.get_cell().volume
        self.assertAlmostEqual(
            cell_initial, cell_relaxed, places=3,
            msg="Cell volume changed during slab relaxation."
        )

        # Check log file is created.
        try:
            with open("test_relax_slab.log", "r") as f:
                log_content = f.read()
            self.assertTrue(len(log_content) > 0, "Slab log file is empty.")
        except FileNotFoundError:
            self.fail("Slab log file was not created.")

    def tearDown(self):
        # Clean up log files if needed.
        import os
        if os.path.exists("test_relax.log"):
            os.remove("test_relax.log")
        if os.path.exists("test_relax_slab.log"):
            os.remove("test_relax_slab.log")


if __name__ == "__main__":
    unittest.main()
