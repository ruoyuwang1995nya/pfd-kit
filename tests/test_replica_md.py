import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import numpy.testing as npt
import os

from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

from pfd.exploration.md.ase_replicas import (
    ReplicaRunner, REMDParameters,
    initialize_replicas,
    attempt_temperature_swap,
    exchange_sweep_oddeven,
    sample_replica_exchange_langevin,
)


class TestReplicaParameters(unittest.TestCase):
    """Test MDParameters dataclass functionality."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_md_parameters_creation(self):
        """Test MDParameters creation with defaults."""
        params = REMDParameters()
        self.assertEqual(params.fmax, 0.01)
        self.assertEqual(params.max_relax_steps, 2000)
        self.assertEqual(params.relax_is_slab, True)
        self.assertEqual(params.min_temperature, 300.0)
        self.assertEqual(params.max_temperature, 1000.0)
        self.assertEqual(params.num_temperature_steps, 5)
        self.assertEqual(params.timestep_fs, 2.0)
        self.assertEqual(params.friction_fs, 0.02)
        self.assertEqual(params.num_cycles, 200)
        self.assertEqual(params.md_steps_per_cycle, 250)
        self.assertEqual(params.log_every_cycles, 10)
        self.assertEqual(params.drop_start_fraction, 0.2)
        self.assertEqual(params.num_samples, 20)
        self.assertIsNone(params.seed)

    def test_md_parameters_custom(self):
        """Test MDParameters with custom values."""
        params = REMDParameters(
            fmax=0.02,
            max_relax_steps=1000,
            relax_is_slab=False,
            min_temperature=400.0,
            max_temperature=800.0,
            num_temperature_steps=3,
            timestep_fs=1.0,
            friction_fs=0.05,
            num_cycles=100,
            md_steps_per_cycle=100,
            log_every_cycles=5,
            drop_start_fraction=0.1,
            num_samples=10,
            seed=42
        )
        self.assertEqual(params.fmax, 0.02)
        self.assertEqual(params.max_relax_steps, 1000)
        self.assertEqual(params.relax_is_slab, False)
        self.assertEqual(params.min_temperature, 400.0)
        self.assertEqual(params.max_temperature, 800.0)
        self.assertEqual(params.num_temperature_steps, 3)
        self.assertEqual(params.timestep_fs, 1.0)
        self.assertEqual(params.friction_fs, 0.05)
        self.assertEqual(params.num_cycles, 100)
        self.assertEqual(params.md_steps_per_cycle, 100)
        self.assertEqual(params.log_every_cycles, 5)
        self.assertEqual(params.drop_start_fraction, 0.1)
        self.assertEqual(params.num_samples, 10)
        self.assertEqual(params.seed, 42)

    def test_md_parameters_json_serialization(self):
        """Test JSON serialization and deserialization."""
        params = REMDParameters(min_temperature=400.0, num_samples=500)

        # Test to_json
        json_str = params.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("400.0", json_str)
        self.assertIn("500", json_str)

        # Test from_json
        params_restored = REMDParameters.from_json(json_str)
        self.assertEqual(params_restored.min_temperature, 400.0)
        self.assertEqual(params_restored.num_samples, 500)
        self.assertEqual(params_restored.friction_fs, 0.02)  # Default value

    def test_md_parameters_file_io(self):
        """Test file I/O operations."""
        params = REMDParameters(max_temperature=1200.0, max_relax_steps=1000)
        json_file = self.test_dir / "params.json"

        # Write to file
        with open(json_file, 'w') as f:
            f.write(params.to_json())

        # Read from file
        params_loaded = REMDParameters.from_file(json_file)
        self.assertEqual(params_loaded.max_temperature, 1200.0)
        self.assertEqual(params_loaded.max_relax_steps, 1000)


# Test utility functions used in ase_replicas.py.
class TestReplicaUtilityFunctions(unittest.TestCase):
    def setUp(self):
        # Create simple atoms for testing
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (1, 1, 1)
        self.calc = EMT()

    def test_initialize_replicas(self):
        """Test initialize_replicas function."""
        num_replicas = 4
        min_temp = 300.0
        max_temp = 600.0

        replicas = initialize_replicas(
            self.atoms,
            self.calc,
            min_temp,
            max_temp,
            num_replicas,
            rng=np.random.default_rng(42),
            log_file="init_replicas.log"
        )

        # Check temperatures correctly set.
        self.assertEqual(len(replicas), num_replicas)
        temps = [replica['T'] for replica in replicas]
        self.assertTrue(all(min_temp <= T <= max_temp for T in temps))

        for rid, replica in enumerate(replicas):
            # Check dynamics correctly set.
            self.assertIn('dyn', replica)
            dyn = replica['dyn']
            self.assertIsNotNone(dyn)
            # Currently only Langevin is implemented.
            self.assertIsInstance(dyn, Langevin)
            dyn_dict = dyn.todict()
            self.assertAlmostEqual(dyn_dict['temperature_K'], replica['T'], places=6)
            self.assertAlmostEqual(dyn_dict['friction'], 0.02 / units.fs, places=6)
            # Check atoms copied correctly.
            a = replica['atoms']
            self.assertIsInstance(a, Atoms)
            npt.assert_allclose(a.get_positions(wrap=True), self.atoms.get_positions(wrap=True), atol=1e-6)
            self.assertEqual(a.get_chemical_symbols(), self.atoms.get_chemical_symbols())
            # Check atom calculator set correctly.
            self.assertIsNotNone(a.calc)
            self.assertEqual(type(a.calc), type(self.calc))
            # Check atom momenta already initialized.
            momenta = a.get_momenta()
            self.assertIsNotNone(momenta)
            # Check index set correctly.
            self.assertEqual(replica['init_id'], rid)

        # Check no master log file created yet.
        self.assertFalse(Path("init_replicas.log").exists())
        # Log file already initialized, once open.
        for rid in range(len(replicas)):
            log_file = f"init_replicas.replica{rid}.log"
            self.assertTrue(Path(log_file).exists())

    def test_attempt_temperature_swap(self):
        """Test attempt_temperature_swap function."""
        # Create two replicas with known energies
        a1 = self.atoms.copy()
        a2 = self.atoms.copy()

        replica1 = {
            'T': 300.0,
            'atoms': a1,
            'init_id': 0,
            'dyn': Langevin(
                a1, timestep=1.0 * units.fs, temperature_K=300.0, friction=0.02 / units.fs
            ),
        }
        replica2 = {
            'T': 600.0,
            'atoms': a2,
            'init_id': 1,
            'dyn': Langevin(
                a2, timestep=1.0 * units.fs, temperature_K=600.0, friction=0.02 / units.fs
            ),
        }

        # Assign energies
        replica1['atoms'].calc = self.calc
        replica2['atoms'].calc = self.calc
        energy1 = replica1['atoms'].get_potential_energy()
        energy2 = replica2['atoms'].get_potential_energy()

        # Force a swap by manipulating energies
        npt.assert_allclose(energy1, energy2, atol=1e-6)

        # Attempt swap
        swapped = attempt_temperature_swap(replica1, replica2, rng=np.random.default_rng(42))

        # Since energies are equal, swap should occur
        self.assertTrue(swapped)
        self.assertEqual(replica1['T'], 600.0)
        self.assertEqual(replica2['T'], 300.0)
        rep1_dict = replica1['dyn'].todict()
        rep2_dict = replica2['dyn'].todict()
        self.assertEqual(rep1_dict['temperature_K'], 600.0)
        self.assertEqual(rep2_dict['temperature_K'], 300.0)
        # All other should not change.
        self.assertEqual(replica1['init_id'], 0)
        self.assertEqual(replica2['init_id'], 1)

    def test_exchange_sweep_oddeven(self):
        """Test exchange_sweep_oddeven function."""
        # Create 3 replicas with identical atoms but different temperatures.
        replicas = [
            {
                'T': 300.0 + i * 100.0,
                'atoms': self.atoms.copy(),
                'init_id': i,
                'dyn': Langevin(
                    self.atoms.copy(), timestep=1.0 * units.fs,
                    temperature_K=300.0 + i * 100.0,
                    friction=0.02 / units.fs
                ),
            }
            for i in range(3)
        ]
        # Set calculators.
        for replica in replicas:
            replica['atoms'].calc = self.calc

        # Attempt odd cycle sweep. Since all energies are equal,
        # swaps should occur between replica index 1 and 2.
        num_swaps, num_attempts = exchange_sweep_oddeven(
            replicas, cycle=11, rng=np.random.default_rng(123)
        )
        self.assertEqual(num_swaps, 1)
        self.assertEqual(num_attempts, 1)
        # Sort happens in sample function, so no change now.
        indices_now = [replica['init_id'] for replica in replicas]
        self.assertEqual(indices_now, [0, 2, 1])  # Replica 1 and 2 swapped.
        Ts_now = [replica['T'] for replica in replicas]
        npt.assert_allclose(Ts_now, [300.0, 400.0, 500.0], atol=1e-6)  # Temperature sorted.

        ## Attempt even cycle sweep. Swap should occur between replica index 0 and 1.
        num_swaps, num_attempts = exchange_sweep_oddeven(
            replicas, cycle=0, rng=np.random.default_rng(123)
        )
        self.assertEqual(num_swaps, 1)
        self.assertEqual(num_attempts, 1)
        indices_now = [replica['init_id'] for replica in replicas]
        self.assertEqual(indices_now, [2, 0, 1])
        Ts_now = [replica['T'] for replica in replicas]
        npt.assert_allclose(Ts_now, [300.0, 400.0, 500.0], atol=1e-6)  # Temperature sorted.

    def test_sample_replica_exchange_langevin(self):
        """Test sample_replica_exchange_langevin function."""
        num_replicas = 3
        min_temp = 300.0
        max_temp = 500.0

        sample = sample_replica_exchange_langevin(
            self.atoms,
            self.calc,
            min_temp,
            max_temp,
            num_replicas,
            num_cycles=20,
            md_steps_per_cycle=50,
            drop_start_fraction=0.2,
            num_samples=4,
            seed=56789,
            log_file="sample_remd.log"
        )

        # Check samples collected.
        self.assertEqual(len(sample), 4)
        for s in sample:
            self.assertIsInstance(s, Atoms)
            self.assertEqual(s.get_chemical_symbols(), self.atoms.get_chemical_symbols())
            # NVT ensemble, cell should not change.
            npt.assert_allclose(s.get_cell(), self.atoms.get_cell(), atol=1e-4)

        # Check log files have all been created.
        self.assertFalse(Path("sample_remd.log").exists())
        for rid in range(num_replicas):
            log_file = f"sample_remd.replica{rid}.log"
            self.assertTrue(Path(log_file).exists())

    def tearDown(self):
        # Remove all created log files.
        for file in Path('.').glob('*.log'):
            file.unlink()



class TestReplicaRunner(unittest.TestCase):
    """Test ReplicaRunner functionality with EMT calculator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)

        # Create test atoms - simple copper cluster
        self.atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        self.atoms = self.atoms * (2, 2, 2)  # 32 atoms

        # Perturb positions slightly to make dynamics interesting
        np.random.seed(42)
        positions = self.atoms.get_positions()
        positions += np.random.normal(0, 0.1, positions.shape)
        self.atoms.set_positions(positions)

        self.params = REMDParameters(
            min_temperature=300.0,
            max_temperature=600.0,
            num_temperature_steps=4,
            num_cycles=20,
            md_steps_per_cycle=10,
            num_samples=4,
            seed=12345,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_runner_creation(self):
        """Test ReplicaRunner creation from atoms."""
        re_runner = ReplicaRunner(self.atoms.copy())
        self.assertEqual(len(re_runner), 32)
        self.assertIsInstance(re_runner.atoms, Atoms)

    def test_runner_from_atoms(self):
        """Test ReplicaRunner.from_atoms class method."""
        re_runner = ReplicaRunner.from_atoms(self.atoms.copy())
        self.assertEqual(len(re_runner), 32)

    def test_runner_from_file(self):
        """Test ReplicaRunner.from_file class method."""
        # Write atoms to file
        structure_file = self.test_dir / "test_structure.cif"
        self.atoms.write(structure_file)

        # Create ReplicaRunner from file
        re_runner = ReplicaRunner.from_file(structure_file)
        self.assertEqual(len(re_runner), 32)
        self.assertEqual(
            re_runner.atoms.get_chemical_symbols(),
            self.atoms.get_chemical_symbols()
        )
        npt.assert_allclose(
            re_runner.atoms.get_positions(wrap=True),
            self.atoms.get_positions(wrap=True),
        )

    def test_calculator_setting(self):
        """Test calculator setting and property access."""
        re_runner = ReplicaRunner(self.atoms.copy())
        calc = EMT()

        # Test set_calculator method
        re_runner.set_calculator(calc)
        self.assertIsInstance(re_runner.calc, EMT)

        # Test calc property setter
        calc2 = EMT()
        re_runner.calc = calc2
        self.assertIs(re_runner.calc, calc2)

    def test_md_simulation(self):
        """Test short NVT MD simulation with EMT."""
        re_runner = ReplicaRunner(self.atoms.copy())
        # Run simulation
        original_dir = Path.cwd()

        # Test run when no calculator set, should raise error
        try:
            os.chdir(self.test_dir)

            with self.assertRaises(ValueError):
                re_runner.run_md(
                    REMDParameters(),
                    log_file="should_not_create.log",
                    traj_file="should_not_create.traj"
                )
            # Should not have created any files
            self.assertFalse(Path("should_not_create.log").exists())
            self.assertFalse(Path("should_not_create.traj").exists())
        finally:
            os.chdir(original_dir)

        # Set EMT calculator
        calc = EMT()
        re_runner.set_calculator(calc)

        # Create short simulation parameters
        params = self.params

        try:
            os.chdir(self.test_dir)

            re_runner.run_md(
                params,
                log_file="test_remd.log",
                traj_file="test_remd.traj"
            )

            # Check traj file length matches expected samples
            from ase.io import read
            traj = read("test_remd.traj", index=":")
            self.assertEqual(len(traj), params.num_samples)
            # Check log file created
            self.assertTrue(Path("test_remd.log").exists())
            # Check log file is concatenated from all replica logs.
            with open("test_remd.log", 'r') as f:
                log_content = f.read()
                for i in range(4):
                    replica_log_filename = f"test_remd.replica{i}.log"
                    self.assertIn(replica_log_filename, log_content)
        finally:
            os.chdir(original_dir)


if __name__ == '__main__':
    unittest.main()
