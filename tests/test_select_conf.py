import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write, read
from dflow.python import OPIO
from pfd.op.select_confs import SelectConfs
from pfd.exploration.selector import ConfSelector
class DummyConfSelector(ConfSelector):
    def select(self, trajs, optional_outputs=None):  # match base signature loosely
        confs = []
        for t in trajs:
            confs.extend(read(t, index=":"))
        return confs

class TestSelectConfs(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tmpdir = tempfile.mkdtemp()
        self.traj_path = Path(self.tmpdir) / "traj1.extxyz"
        atoms_list = [Atoms("H2O", positions=np.random.rand(3,3)) for _ in range(5)]
        write(self.traj_path, atoms_list, format="extxyz")
        self.atoms_list = atoms_list

    # helper to write extxyz with n structures
    def _write_structs(self, filename: str, n: int, element: str = "H"):  # returns path, list
        path = Path(self.tmpdir) / filename
        atoms_list = [Atoms(element, positions=np.random.rand(1,3)) for _ in range(n)]
        write(path, atoms_list, format="extxyz")
        return path, atoms_list

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_select_confs_basic(self):
        op = SelectConfs()
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path],
            "optional_parameters": {"max_sel": 3},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        self.assertLessEqual(len(selected), 3)

    def test_filter_by_entropy_cpu(self):
        op = SelectConfs()
        atoms_list = self.atoms_list
        filtered = op.filter_by_entropy(atoms_list, reference=[], chunk_size=2, max_sel=4, k=2, cutoff=2.0, batch_size=2, h=0.01)
        self.assertIsInstance(filtered, list)
        self.assertGreaterEqual(len(filtered), 1)
        self.assertLessEqual(len(filtered), 5)

    def test_filter_by_entropy_with_reference(self):
        op = SelectConfs()
        ref_path = Path(self.tmpdir) / "ref.extxyz"
        write(ref_path, self.atoms_list[:2], format="extxyz")
        reference = list(read(ref_path, index=":"))
        filtered = op.filter_by_entropy(self.atoms_list, reference=reference, chunk_size=2, max_sel=3, k=2, cutoff=2.0, batch_size=1000, h=0.01)
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), 4)

    def test_select_confs_with_init_confs_entropy(self):
        op = SelectConfs()
        init_path1, init_atoms1 = self._write_structs("init1.extxyz", 2)
        init_path2, init_atoms2 = self._write_structs("init2.extxyz", 1)
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path],
            "init_confs": [init_path1, init_path2],
            "optional_parameters": {"max_sel": 4, "h_filter": {"chunk_size":1, "k":2, "cutoff":2.0, "batch_size":2, "h":0.01}},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        self.assertLessEqual(len(selected), 4)
        self.assertGreaterEqual(len(selected), 1)

    def test_select_confs_with_iter_confs_entropy(self):
        op = SelectConfs()
        iter_path, iter_atoms = self._write_structs("iter.extxyz", 3)
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path],
            "iter_confs": iter_path,
            "optional_parameters": {"max_sel": 5, "h_filter": {"chunk_size":1, "k":2, "cutoff":2.0, "batch_size":2, "h":0.01}},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        self.assertLessEqual(len(selected), 5)
        self.assertGreaterEqual(len(selected), 1)

    def test_select_confs_with_both_init_iter_entropy(self):
        op = SelectConfs()
        init_path1, _ = self._write_structs("initb1.extxyz", 2)
        iter_path, _ = self._write_structs("iterb.extxyz", 2)
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path],
            "init_confs": [init_path1],
            "iter_confs": iter_path,
            "optional_parameters": {"max_sel": 5, "h_filter": {"chunk_size":1, "k":2, "cutoff":2.0, "batch_size":2, "h":0.01}},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        self.assertLessEqual(len(selected), 5)
        self.assertGreaterEqual(len(selected), 1)

    def test_select_confs_with_directories(self):
        """Test that SelectConfs can handle directories containing trajectory files."""
        op = SelectConfs()
        
        # Create subdirectories with trajectory files
        subdir1 = Path(self.tmpdir) / "subdir1"
        subdir2 = Path(self.tmpdir) / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()
        
        # Create trajectory files in subdirectories
        traj1_path = subdir1 / "traj1.extxyz"
        traj2_path = subdir1 / "traj2.traj"
        traj3_path = subdir2 / "traj3.extxyz"
        
        atoms_list1 = [Atoms("H2O", positions=np.random.rand(3,3)) for _ in range(3)]
        atoms_list2 = [Atoms("CO2", positions=np.random.rand(3,3)) for _ in range(2)]
        atoms_list3 = [Atoms("NH3", positions=np.random.rand(4,3)) for _ in range(4)]
        
        write(traj1_path, atoms_list1, format="extxyz")
        write(traj2_path, atoms_list2, format="traj")
        write(traj3_path, atoms_list3, format="extxyz")
        
        # Test with directories as input
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [subdir1, subdir2],  # Pass directories instead of files
            "optional_parameters": {"max_sel": 8},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        
        # Should have found all atoms from both directories
        # Total: 3 + 2 + 4 = 9 atoms, but limited by max_sel=8
        self.assertLessEqual(len(selected), 8)
        self.assertGreaterEqual(len(selected), 1)

    def test_select_confs_mixed_files_and_directories(self):
        """Test that SelectConfs can handle a mix of files and directories."""
        op = SelectConfs()
        
        # Create a subdirectory with trajectory files
        subdir = Path(self.tmpdir) / "mixed_test"
        subdir.mkdir()
        
        # Create trajectory file in subdirectory
        traj_in_dir = subdir / "traj_in_dir.extxyz"
        atoms_in_dir = [Atoms("He", positions=np.random.rand(1,3)) for _ in range(2)]
        write(traj_in_dir, atoms_in_dir, format="extxyz")
        
        # Test with mix of file and directory
        ip = OPIO({
            "conf_selector": DummyConfSelector(),
            "confs": [self.traj_path, subdir],  # Mix of file and directory
            "optional_parameters": {"max_sel": 10},
        })
        out = op.execute(ip)
        out_path = out["confs"]
        self.assertTrue(Path(out_path).exists())
        selected = list(read(out_path, index=":"))
        
        # Should have found atoms from both the direct file and the directory
        # Total: 5 (from self.traj_path) + 2 (from subdir) = 7 atoms
        self.assertLessEqual(len(selected), 10)
        self.assertGreaterEqual(len(selected), 1)

    def test_expand_directories_method(self):
        """Test the _expand_directories method directly."""
        op = SelectConfs()
        
        # Create test directory structure
        test_dir = Path(self.tmpdir) / "expand_test"
        test_dir.mkdir()
        
        # Create various file types
        traj_file = test_dir / "valid.traj"
        xyz_file = test_dir / "valid.xyz"
        extxyz_file = test_dir / "valid.extxyz"
        txt_file = test_dir / "invalid.txt"
        
        # Create some dummy content
        atoms = Atoms("H", positions=[[0,0,0]])
        write(traj_file, atoms, format="traj")
        write(xyz_file, atoms, format="xyz")
        write(extxyz_file, atoms, format="extxyz")
        txt_file.write_text("not a trajectory file")
        
        # Test expansion
        input_paths = [self.traj_path, test_dir]  # Mix of file and directory
        expanded = op._expand_directories(input_paths)
        
        # Should include the original file plus the valid trajectory files from the directory
        expanded_names = [p.name for p in expanded]
        self.assertIn("traj1.extxyz", expanded_names)  # Original file
        self.assertIn("valid.traj", expanded_names)    # From directory
        self.assertIn("valid.xyz", expanded_names)     # From directory
        self.assertIn("valid.extxyz", expanded_names)  # From directory
        self.assertNotIn("invalid.txt", expanded_names) # Should be filtered out

if __name__ == "__main__":
    unittest.main()
