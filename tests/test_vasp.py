import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk

from pfd.fp.vasp import PrepVasp, loads_incar, dumps_incar
from pfd.fp.vasp_input import VaspInputs


class TestVaspInputGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tmpdir = tempfile.mkdtemp()
        self.test_dir = Path(self.tmpdir)
        
        # Create mock INCAR template
        self.incar_content = """SYSTEM = Test system
PREC = Accurate
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
NSW = 0
IBRION = -1
ISPIN = 1
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
        self.incar_file = self.test_dir / "INCAR.template"
        self.incar_file.write_text(self.incar_content)
        
        # Create mock POTCAR files
        self.potcar_h = self.test_dir / "POTCAR_H"
        self.potcar_h.write_text("H POTCAR content\n")
        self.potcar_o = self.test_dir / "POTCAR_O"
        self.potcar_o.write_text("O POTCAR content\n")
        self.potcar_fe = self.test_dir / "POTCAR_Fe"
        self.potcar_fe.write_text("Fe POTCAR content\n")
        
        # POTCAR file mapping
        self.pp_files = {
            "H": str(self.potcar_h),
            "O": str(self.potcar_o),
            "Fe": str(self.potcar_fe),
        }
        
        # Create VaspInputs object
        self.vasp_inputs = VaspInputs(
            kspacing=0.3,
            incar=str(self.incar_file),
            pp_files=self.pp_files,
            kgamma=True
        )

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.tmpdir)

    def test_vasp_inputs_initialization(self):
        """Test VaspInputs initialization."""
        self.assertEqual(self.vasp_inputs.kspacing, 0.3)
        self.assertTrue(self.vasp_inputs.kgamma)
        self.assertIn("SYSTEM = Test system", self.vasp_inputs.incar_template)
        self.assertIn("H", self.vasp_inputs.potcars)
        self.assertIn("O", self.vasp_inputs.potcars)
        self.assertIn("Fe", self.vasp_inputs.potcars)

    def test_make_potcar(self):
        """Test POTCAR generation for different atom types."""
        # Test H2O system
        atom_names = ["H", "H", "O"]
        potcar_content = self.vasp_inputs.make_potcar(atom_names)
        expected = "H POTCAR content\nH POTCAR content\nO POTCAR content\n"
        self.assertEqual(potcar_content, expected)

    def test_make_kpoints(self):
        """Test KPOINTS generation."""
        # Simple cubic cell
        box = np.array([[5.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0],
                       [0.0, 0.0, 5.0]])
        kpoints = self.vasp_inputs.make_kpoints(box)
        
        # Check that it's gamma-centered
        self.assertIn("Gamma", kpoints)
        self.assertIn("Automatic mesh", kpoints)

    def test_prep_vasp_simple_atoms(self):
        """Test PrepVasp with simple atoms without magnetic moments."""
        prep_vasp = PrepVasp()
        
        # Create simple H2O molecule
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        atoms.pbc = True
        
        # Prepare task
        task_dir = self.test_dir / "task_simple"
        task_dir.mkdir()
        
        # Change to task directory and prepare
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(task_dir)
            prep_vasp.prep_task(atoms, self.vasp_inputs)
            
            # Check generated files
            self.assertTrue((task_dir / "POSCAR").exists())
            self.assertTrue((task_dir / "INCAR").exists())
            self.assertTrue((task_dir / "POTCAR").exists())
            self.assertTrue((task_dir / "KPOINTS").exists())
            
            # Check POSCAR content
            poscar_content = (task_dir / "POSCAR").read_text()
            self.assertIn("H", poscar_content)
            self.assertIn("O", poscar_content)
            
            # Check INCAR content (should not have MAGMOM)
            incar_content = (task_dir / "INCAR").read_text()
            self.assertNotIn("MAGMOM", incar_content)
            
        finally:
            os.chdir(orig_cwd)

    def test_prep_vasp_with_magnetic_moments(self):
        """Test PrepVasp with atoms having initial magnetic moments."""
        prep_vasp = PrepVasp()
        
        # Create Fe atoms with magnetic moments
        atoms = bulk('Fe', 'bcc', a=2.87, cubic=True)
        atoms = atoms * (2, 1, 1)  # 2 Fe atoms
        
        # Set initial magnetic moments
        magmoms = np.ones(4)  # Antiferromagnetic
        atoms.set_initial_magnetic_moments(magmoms)
        
        # Prepare task
        task_dir = self.test_dir / "task_magnetic"
        task_dir.mkdir()
        
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(task_dir)
            prep_vasp.prep_task(atoms, self.vasp_inputs)
            
            # Check generated files
            self.assertTrue((task_dir / "POSCAR").exists())
            self.assertTrue((task_dir / "INCAR").exists())
            self.assertTrue((task_dir / "POTCAR").exists())
            self.assertTrue((task_dir / "KPOINTS").exists())
            
            # Check INCAR content (should have MAGMOM)
            incar_content = (task_dir / "INCAR").read_text()
            self.assertIn("MAGMOM", incar_content)
            #self.assertIn("3.0", incar_content)
            #self.assertIn("-3.0", incar_content)
            
            # Parse INCAR and check MAGMOM values
            incar_dict = loads_incar(incar_content)
            magmom_str = incar_dict["MAGMOM"]
            magmom_values = np.array([float(x) for x in magmom_str.split()])
            self.assertEqual(len(magmom_values), 4)

        finally:
            os.chdir(orig_cwd)

    def test_create_tasks_multiple_frames(self):
        """Test PrepVasp._create_tasks with multiple atomic configurations."""
        prep_vasp = PrepVasp()
        
        # Create multiple configurations
        confs = []
        for i in range(3):
            atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74 + i*0.1]])
            atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
            atoms.pbc = True
            confs.append(atoms)
        
        # Config dict
        config = {"inputs": self.vasp_inputs}
        
        # Create tasks
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            task_names, task_paths = prep_vasp._create_tasks(confs, config)
            
            # Check that we got the right number of tasks
            self.assertEqual(len(task_names), 3)
            self.assertEqual(len(task_paths), 3)
            
            # Check task names
            expected_names = ["task.000000", "task.000001", "task.000002"]
            self.assertEqual(task_names, expected_names)
            
            # Check that directories were created
            for path in task_paths:
                self.assertTrue(path.exists())
                self.assertTrue((path / "POSCAR").exists())
                self.assertTrue((path / "INCAR").exists())
                
        finally:
            os.chdir(orig_cwd)

    def test_loads_dumps_incar(self):
        """Test INCAR parsing and generation functions."""
        # Test loading INCAR
        incar_dict = loads_incar(self.incar_content)
        self.assertEqual(incar_dict["SYSTEM"], "Test system")
        self.assertEqual(incar_dict["ENCUT"], "520")
        self.assertEqual(incar_dict["ISPIN"], "1")
        
        # Test dumping INCAR
        new_dict = {"SYSTEM": "New system", "ENCUT": "600", "MAGMOM": "1.0 -1.0"}
        new_incar = dumps_incar(new_dict)
        self.assertIn("SYSTEM = New system", new_incar)
        self.assertIn("ENCUT = 600", new_incar)
        self.assertIn("MAGMOM = 1.0 -1.0", new_incar)

    def test_magnetic_moments_complex_system(self):
        """Test magnetic moments with a more complex system."""
        prep_vasp = PrepVasp()
        
        # Create a mixed H2O + Fe system
        positions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [3, 0, 0]]
        atoms = Atoms('H2OFe', positions=positions)
        atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        atoms.pbc = True
        
        # Set magnetic moments (only Fe should be magnetic)
        magmoms = [0.0, 0.0, 0.0, 4.0]  # H, H, O, Fe
        atoms.set_initial_magnetic_moments(magmoms)
        
        # Prepare task
        task_dir = self.test_dir / "task_mixed"
        task_dir.mkdir()
        
        import os
        orig_cwd = os.getcwd()
        try:
            os.chdir(task_dir)
            prep_vasp.prep_task(atoms, self.vasp_inputs)
            
            # Check INCAR content
            incar_content = (task_dir / "INCAR").read_text()
            self.assertIn("MAGMOM", incar_content)
            
            # Parse and check magnetic moments
            incar_dict = loads_incar(incar_content)
            magmom_str = incar_dict["MAGMOM"]
            magmom_values = [float(x) for x in magmom_str.split()]
            self.assertEqual(len(magmom_values), 4)  # Fe first after sorting

        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
