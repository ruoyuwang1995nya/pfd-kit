import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write, read
from pfd.op.select_confs import SelectConfs
from pfd.exploration.selector import ConfSelectorFrames,ConfSelector
class DummyConfSelector(ConfSelector):
    def select(self, trajs):
        # Just read all atoms from all files
        confs = []
        for t in trajs:
            confs.extend(read(t, index=":"))
        return confs

class TestSelectConfs(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for test files
        self.tmpdir = tempfile.mkdtemp()
        self.traj_path = Path(self.tmpdir) / "traj1.extxyz"
        # Generate 5 simple Atoms objects and write to extxyz
        atoms_list = [Atoms("H2O", positions=np.random.rand(3,3)) for _ in range(5)]
        write(self.traj_path, atoms_list, format="extxyz")
        self.atoms_list = atoms_list

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_select_confs_basic(self):
        op = SelectConfs()
        ip = {
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path],
            "optional_parameters": {"max_sel": 3},
        }
        out = op.execute(ip)
        # Output file should exist
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        # Should have at most max_sel structures
        selected = list(read(out_path, index=":"))
        self.assertLessEqual(len(selected), 3)

    def test_filter_by_entropy_cpu(self):
        op = SelectConfs()
        # Use a small set of Atoms objects
        atoms_list = self.atoms_list
        # Should not raise and should return a non-empty list
        filtered = op.filter_by_entropy(atoms_list, reference=None, chunk_size=2, max_sel=4, k=2, cutoff=2.0, batch_size=2, h=0.01)
        self.assertIsInstance(filtered, list)
        self.assertGreaterEqual(len(filtered), 2)
        self.assertLessEqual(len(filtered), 5)

    def test_filter_by_entropy_with_reference(self):
        op = SelectConfs()
        # Write a reference extxyz file
        ref_path = Path(self.tmpdir) / "ref.extxyz"
        write(ref_path, self.atoms_list[:2], format="extxyz")
        filtered = op.filter_by_entropy(self.atoms_list, reference=ref_path, chunk_size=2, max_sel=3, k=2, cutoff=2.0, batch_size=2, h=0.01)
        self.assertIsInstance(filtered, list)
        self.assertGreaterEqual(len(filtered), 2)
        self.assertLessEqual(len(filtered), 7)

if __name__ == "__main__":
    unittest.main()
