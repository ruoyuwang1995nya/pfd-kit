from typing import List, Optional
from ase import Atoms
from ase.io import read, write
from pathlib import Path
import numpy as np

def filter_by_entropy(
        confs: List[Atoms],
        reference: Optional[Path]=None,
        k=32,
        cutoff=5.0,
        batch_size: int = 1000,
        h = 0.015,
        entropy_threshold: float = 0.01,
        **kwargs
        )-> List[Atoms]:
        """Filter structures to maximize entropy/diversity."""
        from quests.descriptor import get_descriptors
        from tqdm import tqdm
        def create_entropy_function():
            """Factory function to create the appropriate entropy function."""
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                from quests.gpu.entropy import entropy as gpu_entropy
        
                def get_entropy(x: np.ndarray, **kwargs):
                    x_tensor = torch.from_numpy(x)
                    return gpu_entropy(x_tensor, device=device, **kwargs)
            
                logger.info(f"Using GPU entropy with device: {device}")
                return get_entropy
        
            except ImportError:
                from quests.entropy import entropy as cpu_entropy
                def get_entropy(x: np.ndarray, **kwargs):
                    return cpu_entropy(x, **kwargs)
                logger.info("Using CPU entropy (torch not available)")
                return get_entropy
            
        num_confs=len(confs)
        get_entropy = create_entropy_function()
        filtered_structures = []
            
        if reference is not None:
            reference = read(reference, index=":")
        else:
            n_ref = max(1, min(100, len(confs) // 10))
            ref_indices = np.random.choice(len(confs), n_ref, replace=False)
            reference = [confs[i] for i in ref_indices]
            write('text.extxyz', reference, format='extxyz')
            other_indices = np.setdiff1d(np.arange(len(confs)), ref_indices)
            confs = [confs[i] for i in other_indices]
        current_descriptors = get_descriptors(reference,k=k,cutoff=cutoff)
        for atoms in tqdm(confs):
            cand_desc = get_descriptors([atoms],k=k,cutoff=cutoff)
            current_entropy = get_entropy(
                current_descriptors, 
                h=h,
                batch_size=batch_size,
            )
            tmp_descriptors = np.vstack([current_descriptors, cand_desc])
            entropy_tmp = get_entropy(
                tmp_descriptors, 
                h=0.015, batch_size=10000
            )
            entropy_delta = entropy_tmp - current_entropy
            if entropy_delta > entropy_threshold:
                filtered_structures.append(atoms)
                current_descriptors = tmp_descriptors
        logger.info(f"Entropy filtering: selected {len(filtered_structures)} structures from {num_confs} candidates.")
        logger.info(f"Entropy: {entropy_tmp}")
        return filtered_structures


# ...existing code...
import torch
from quests.gpu.entropy import entropy, delta_entropy
from quests.descriptor import get_descriptors
import numpy as np
from ase.io import read, write

h = 0.015
batch_size = 10000
iter_dset = read('./iter2.extxyz', index=":")
tot_h = 9.2
chunk_size = 100
indices = []
dset = read('./iter1/1650.extxyz', index=":")
init_chunk_size = len(dset)

desc_iter = get_descriptors(iter_dset, k=32, cutoff=5.0)
desc_dset = get_descriptors(dset, k=32, cutoff=5.0)

# Compute atom indices for iter_dset
num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_dset]
atom_indices_iter = []
start = 0
for n in num_atoms_per_structure_iter:
    end = start + n
    atom_indices_iter.append((start, end))
    start = end

h_ls=[]
for ii in range(20):
    re_indices = [i for i in range(len(iter_dset)) if i not in indices]
    re_dset = [iter_dset[i] for i in re_indices]
    re_desc = [desc_iter[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
    x = torch.tensor(np.vstack(re_desc), device="cuda", dtype=torch.float32)
    y = torch.tensor(desc_dset, device="cuda", dtype=torch.float32)
    delta = delta_entropy(x, y, h=h,batch_size=batch_size)
    delta_arr = delta.cpu().numpy()
    num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_dset]
    atom_indices = []
    start = 0
    for n in num_atoms_per_structure:
        end = start + n
        atom_indices.append((start, end))
        start = end
    delta_sums = [delta_arr[start:end].sum() for start, end in atom_indices]
    sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
    sorted_re_indices = [idx for idx, _ in sorted_pairs]
    
    #sorted_indices = sorted(range(len(delta_sums)), key=lambda i: delta_sums[i], reverse=True)
    selected_indices = sorted_re_indices[:chunk_size]
    desc_dset_ls=[desc_dset]
    for idx in selected_indices:
        indices.append(idx)
        dset.append(iter_dset[idx])
        desc_dset_ls.append(desc_iter[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
    desc_dset = np.vstack(desc_dset_ls)
    #print(desc_dset.shape)
    # Recompute y only (dset_descriptors has grown)
    y = torch.tensor(desc_dset, device="cuda", dtype=torch.float32)
    H = entropy(y, h=h, batch_size=batch_size)
    print((ii+1)*chunk_size+init_chunk_size, H)
    write(f"iter1/{(ii+1)*chunk_size+init_chunk_size}.extxyz", dset)
    h_ls.append(float(H.cpu().numpy()))
    if H >= tot_h*1.1:
        break
# ...existing code...