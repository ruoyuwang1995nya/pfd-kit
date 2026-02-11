import numpy as np
import pytest
from ase import Atoms
from pfd.utils.build import perturbed_atoms, perturb

# -------------------------
# 辅助函数：计算所有原子对键长变化百分比
# -------------------------
def bond_length_changes(atoms_orig, atoms_pert):
    """
    返回所有原子对的相对键长变化百分比
    Δd/d0
    """
    pos0 = atoms_orig.get_positions()
    pos1 = atoms_pert.get_positions()
    n = len(atoms_orig)
    changes = []
    for i in range(n):
        for j in range(i+1, n):
            d0 = np.linalg.norm(pos0[i] - pos0[j])
            d1 = np.linalg.norm(pos1[i] - pos1[j])
            if d0 > 1e-8:  # 避免零除
                changes.append((d1 - d0) / d0)
    return np.array(changes)

# -------------------------
# 1. 单胞扰动测试
# -------------------------
def test_single_cell_perturb():
    atoms = Atoms('Si2', positions=[[0,0,0],[0.25,0.25,0.25]], cell=[1,1,1], pbc=True)
    pert_atoms = perturbed_atoms(atoms, pert_num=1, cell_pert_fraction=0.01, atom_pert_distance=0.01)
    for pa in pert_atoms:
        changes = bond_length_changes(atoms, pa)
        assert np.all((changes >= -0.1) & (changes <= 0.1)), \
            f"Single cell bond lengths changed out of -10%~10% range: {changes}"

# -------------------------
# 2. 超胞扰动测试
# -------------------------
def test_supercell_perturb():
    atoms = Atoms('Si2', positions=[[0,0,0],[0.25,0.25,0.25]], cell=[1,1,1], pbc=True)
    atoms_super = atoms * (2,2,2)
    pert_atoms = perturbed_atoms(atoms_super, pert_num=1, cell_pert_fraction=0.01, atom_pert_distance=0.01)
    for pa in pert_atoms:
        changes = bond_length_changes(atoms_super, pa)
        assert np.all((changes >= -0.1) & (changes <= 0.1)), \
            f"Supercell bond lengths changed out of -10%~10% range: {changes}"

# -------------------------
# 3. 扰动风格测试
# -------------------------
@pytest.mark.parametrize("style", ["normal", "uniform", "const"])
def test_atom_perturb_styles(style):
    atoms = Atoms('Si2', positions=[[0,0,0],[0.25,0.25,0.25]], cell=[1,1,1], pbc=True)
    pert_atoms = perturbed_atoms(atoms, pert_num=1, cell_pert_fraction=0.01, atom_pert_distance=0.01, atom_pert_style=style)
    for pa in pert_atoms:
        changes = bond_length_changes(atoms, pa)
        assert np.all((changes >= -0.1) & (changes <= 0.1)), \
            f"Style {style} bond lengths changed out of -10%~10% range: {changes}"

# -------------------------
# 4. 扰动概率测试
# -------------------------
def test_atom_perturb_probability():
    atoms = Atoms('Si5', positions=np.random.rand(5,3), cell=[1,1,1], pbc=True)
    pert_atoms = perturbed_atoms(atoms, pert_num=1, cell_pert_fraction=0.01, atom_pert_distance=0.01, atom_pert_prob=0.5)
    for pa in pert_atoms:
        changes = bond_length_changes(atoms, pa)
        # 至少有一条键长变化不为零
        assert np.any(changes != 0), "No bond lengths were perturbed"

# -------------------------
# 5. 边界条件测试
# -------------------------
def test_zero_perturb():
    atoms = Atoms('Si2', positions=[[0,0,0],[0.25,0.25,0.25]], cell=[1,1,1], pbc=True)
    pert_atoms = perturbed_atoms(atoms, pert_num=1, cell_pert_fraction=0.0, atom_pert_distance=0.0)
    for pa in pert_atoms:
        changes = bond_length_changes(atoms, pa)
        assert np.all(changes == 0), "Bond lengths changed when perturbation was zero"

# -------------------------
# 6. perturb() 接口测试
# -------------------------
def test_perturb_interface():
    atoms = Atoms('Si2', positions=[[0,0,0],[0.25,0.25,0.25]], cell=[1,1,1], pbc=True)
    pert_list = perturb(atoms, pert_num=2, cell_pert_fraction=0.01, atom_pert_distance=0.01, supercell=(2,2,2))
    assert len(pert_list) == 2, "perturb() did not return correct number of perturbed atoms"

