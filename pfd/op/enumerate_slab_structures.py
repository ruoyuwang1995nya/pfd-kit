"""Utility functions and OPs to enumerate slab structures."""
from typing import Optional, List, Dict

from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import math
import warnings
import logging
from logging import Logger


def tasker2_with_odd_num_sites_fallback(
        slab: Slab,
        logger: Optional[Logger] = None,
) -> List[Slab]:
    """Perform Tasker 2 modification with fallback for odd number of sites.

    If number of sites is odd, retry with doubled super-cell.
    Args:
        slab (Slab): The input slab structure.
        logger (Optional[Logger]): Logger for logging messages.
            If None, uses default logger.
    Returns:
        list[Slab]: List of tasker 2 modified slab structures.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    retry = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = slab.get_tasker2_slabs(tol=0.01)
        # No warning, return directly.
        if len(w) == 0:
            return result

        # Scan all warnings.
        for warn in w:
            msg = str(warn.message).lower()

            if "odd number of sites to divide" in msg:
                retry = True
                break

    if retry:
        logger.info("Odd number of sites. Retry with doubled super-cell in x-direction.")
        new_slab = slab.copy()
        new_slab.make_supercell([2, 1, 1])
        return new_slab.get_tasker2_slabs(tol=0.01)

    # Ignore all other warnings.
    return result


def get_slabs(
        prim: Structure,
        miller_index: tuple[int, int, int],
        symprec: float=0.1,
        angle_tol: float=8,
        min_slab_ab: float=12.0,
        min_slab: float=12.0,
        min_vac: float=20.0,
        max_normal_search: int=20,
        symmetrize_slab: bool=False,
        tasker2_modify_polar: bool=True,
        drop_polar: bool=True,
) -> List[Slab]:
    """Generate slabs without vacancies from a primitive structure.

    See pymatgen.core.surface.SlabGenerator for details in arguments max_normal_search,
    symmetrize_slab, etc.

    Args:
        prim (Structure): Primitive structure. Make sure to use symmetric structure to prevent unexpected
            slab generation results. Also, conventional standard cells are recommended.
        miller_index (tuple[int, int, int]): Miller index for slab generation.
        symprec (float, optional): Symmetry precision for SpacegroupAnalyzer. Defaults to 0.1.
        angle_tol (float, optional): Angle tolerance for SpacegroupAnalyzer. Defaults to 8.
        min_slab_ab (float, optional):
            Minimum slab size in a and b directions after supercell construction. Defaults to 12.0.
        min_slab (float, optional): Minimum slab thickness in c direction. Defaults to 12.0.
        min_vac (float, optional): Minimum vacuum thickness in c direction. Defaults to 20.0.
        max_normal_search (int, optional):
            Determines maximum integer supercell factor to search for a normal c direction.
            Defaults to 20.
        symmetrize_slab (bool, optional):
            Whether to symmetrize the slab. Defaults to False as it may lead to high search cost.
        tasker2_modify_polar (bool, optional): Whether to apply Tasker 2 modification to polar slabs.
            Defaults to True as polar slabs often result in unphysical DFT results.
        drop_polar (bool): Whether to drop polar slabs after Tasker modification.
            Defaults to True.
    Returns:
        list[Slab]: List of generated slabs.
    """
    sa = SpacegroupAnalyzer(prim, symprec=symprec, angle_tolerance=angle_tol)
    conv = sa.get_conventional_standard_structure()
    generator = SlabGenerator(
        conv,
        miller_index=miller_index,
        min_slab_size=min_slab,
        min_vacuum_size=min_vac,
        center_slab=True,
        primitive=True,
        max_normal_search=max_normal_search
    )
    slabs = generator.get_slabs(symmetrize=symmetrize_slab)
    final_slabs = []
    if tasker2_modify_polar:
        for slab in slabs:
            if slab.is_polar():
                print("Tasker modification to polar surface.")
                final_slabs.extend(tasker2_with_odd_num_sites_fallback(slab))
            else:
                final_slabs.append(slab)
    else:
        final_slabs = slabs

    # Drop slabs failing tasker correction.
    if drop_polar:
        final_slabs = [slab for slab in final_slabs if not slab.is_polar()]

    # Tasker modification before supercell construction.
    for slab in final_slabs:
        mult_a = int(np.ceil(min_slab_ab / slab.lattice.a))
        mult_b = int(np.ceil(min_slab_ab / slab.lattice.b))
        slab.make_supercell([mult_a, mult_b, 1])
    return final_slabs


# Find mirror/inverse equivalence. Only works for ordered structures (structure with no partial occupancy)
# and structure without overlapping atoms.
def get_mapping_pbc(
        A: np.ndarray,
        B: np.ndarray,
        dtol: float=1e-6
) -> Optional[List[int]]:
    """Get mapping from fractional coordinates A to B under periodic boundary conditions.

    Only works for ordered structures without overlapping atoms.
    Args:
        A (np.ndarray): Nx3 array of fractional coordinates.
        B (np.ndarray): Nx3 array of fractional coordinates.
        dtol (float, optional): Fractional difference tolerance to consider two points equivalent.
            Defaults to 1e-6.
    Returns:
        Optional[list[int]]: Mapping list where mapping[i] gives the index in B that corresponds to A[i].
            Returns None if no valid one-to-one mapping exists.
    """
    if A.shape != B.shape:
        return None
    # N_A * N_B * 3
    dr_tensor = A[:, None, :] - B[None, :, :]
    # Minimal distance under PBC.
    d_matrix = np.linalg.norm(dr_tensor - np.round(dr_tensor), axis=-1)  # Distance in PBC.
    mapping_matrix = np.isclose(d_matrix, 0.0, atol=dtol)
    if np.allclose(mapping_matrix.sum(axis=0), 1.0) and np.allclose(mapping_matrix.sum(axis=1), 1.0):
        # this means to get A, use B[mapping].
        return [int(np.where(mapping_matrix[i])[0][0]) for i in range(len(A))]
    else:
        return None


def get_surface_mapping_pbc(
        arr_a_in: np.ndarray,
        arr_b_in: np.ndarray,
        dtol: float=1e-6,
        mode: str="mirror"
) -> Optional[List[int]]:
    """Get relation mapping from arr_a to arr_b under PBC with mirror or inverse operation.

    Args:
        arr_a_in (np.ndarray): Nx3 array of fractional coordinates for surface A.
        arr_b_in (np.ndarray): Nx3 array of fractional coordinates for surface B.
        dtol (float, optional): Fractional difference tolerance to consider two points equivalent.
            Defaults to 1e-6.
        mode (str, optional): "mirror" or "inverse" operation mode. Defaults to "mirror".
    Returns:
        Optional[list[int]]: Mapping list where mapping[i] gives the index in arr_b that
            corresponds to arr_a[i] under the specified operation.
            Returns None if no valid one-to-one mapping exists.
    """
    arr_a = arr_a_in.copy()
    arr_b = arr_b_in.copy()
    if len(arr_a) != len(arr_b):
        return None
    if mode == "mirror":
        arr_b[:, 2] = -1.0 * arr_b[:, 2]
        anchor_a = arr_a[0]
        # Find anchor of b as the closest point to anchor of a in x, y direction.
        dxy_to_anchor_a = np.linalg.norm(
            (arr_b[:, :2] - anchor_a[:2]) - np.round(arr_b[:, :2] - anchor_a[:2]),
            axis=-1
        )
        anchor_b_inds = np.where(np.isclose(dxy_to_anchor_a, 0.0, atol=dtol))[0]
        if len(anchor_b_inds) == 0:  # No valid anchor.
            return None
        anchor_b = arr_b[anchor_b_inds[0]]
        dz = anchor_a[2] - anchor_b[2]
        tau = np.array([0.0, 0.0, dz])
        arr_b += tau  # b aligned with a.

        mapping = get_mapping_pbc(arr_a, arr_b, dtol=dtol)
        if mapping is not None:
            return mapping
        else:
            return None

    elif mode == "inverse":
        arr_b = -1.0 * arr_b
        anchor_a = arr_a[0]
        for rb in arr_b:
            anchor_b = rb.copy()
            tau = anchor_a - anchor_b  # tau = A - B
            arr_b_cp = arr_b.copy()
            arr_b_cp += tau

            mapping = get_mapping_pbc(arr_a, arr_b_cp, dtol=dtol)
            if mapping is not None:
                return mapping

        # If all anchor has failed, return None.
        return None
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def get_slab_equivalence_groups(
        species: List[str],
        surface_sites_and_indices: Dict[str, List],
        frac_tol: float=1e-6,
        mode: str="mirror"
) -> Optional[Dict[int, int]]:
    """Get equivalence mapping between top and bottom surface sites of a slab.

    Using this mapping, one can symmetrically remove sites from both surfaces to prevent
    extra dipole formation.

    Args:
        species (list[str]): List of species for all sites in the slab.
        surface_sites_and_indices (dict[str, list]): Dictionary with keys "top" and "bottom",
            each containing a list of tuples (site, index) for surface sites.
        frac_tol (float, optional): Fractional coordinate tolerance for mapping. Defaults to 1e-6.
        mode (str, optional): "mirror" or "inverse" operation mode. Defaults to "mirror".
    Returns:
        Optional[dict[int, int]]: Mapping dictionary where keys are indices of top surface sites
            and values are corresponding indices of bottom surface sites.
            Returns None if no valid mapping exists.
    """
    top_sites_and_indices = surface_sites_and_indices["top"]
    bot_sites_and_indices = surface_sites_and_indices["bottom"]
    top_frac_coords = np.array([site.frac_coords for site, ind in top_sites_and_indices])
    top_frac_coords -= np.floor(top_frac_coords)
    top_inds = np.array([ind for site, ind in top_sites_and_indices], dtype=int)
    bot_frac_coords = np.array([site.frac_coords for site, ind in bot_sites_and_indices])
    bot_frac_coords -= np.floor(bot_frac_coords)
    bot_inds = np.array([ind for site, ind in bot_sites_and_indices], dtype=int)

    mapping = get_surface_mapping_pbc(top_frac_coords, bot_frac_coords, dtol=frac_tol, mode=mode)
    if mapping is None:
        return None
    else:
        map_dict = {}
        for i in range(len(mapping)):
            if species[top_inds[i]] == species[bot_inds[mapping[i]]]:
                map_dict[top_inds[i]] = bot_inds[mapping[i]]
            else:
                return None
        return map_dict


def symmetrically_remove_top_sites(
        slab: Slab,
        top_remove_inds: List[int],
        top_to_bottom_mapping: Dict[int, int],
) -> Slab:
    """Symmetrically remove top sites and their corresponding bottom sites from a slab.

    Args:
        slab (Slab): The input slab structure.
        top_remove_inds (list[int]): List of indices of top surface sites to remove.
        top_to_bottom_mapping (dict[int, int]): Mapping from top surface site indices to bottom
            surface site indices.
    Returns:
        Slab: New slab structure with specified sites removed.
    """
    bot_remove_inds = [top_to_bottom_mapping[ii] for ii in top_remove_inds]
    remove_inds = top_remove_inds + bot_remove_inds
    slab_cp = slab.copy()
    slab_cp.remove_sites(remove_inds)
    return slab_cp


def remove_isolated_atoms(
        struct: Structure,
        radius: float=3.0
) -> Structure:
    """Remove isolated atoms from a structure.

    Args:
        struct (Structure): Input structure.
        radius (float, optional): Radius to consider neighbors. Defaults to 3.0.
    Returns:
        Structure: New structure with isolated atoms removed.
    """
    isolated_inds = []
    for i in range(len(struct)):
        if len(struct.get_neighbors(struct[i], radius)) == 0:
            isolated_inds.append(i)
    struct_cp = struct.copy()
    struct_cp.remove_sites(isolated_inds)
    return struct_cp

# TODO: write docs and tests for the following functions.
def get_site_charge(site):
    if isinstance(site.specie, Element):
        return 0
    else:
        return site.specie.oxi_state


def get_z_range_indices(structure, zmin, zmax):
    frac_coords = structure.frac_coords
    frac_coords = frac_coords - np.floor(frac_coords)
    return np.where((frac_coords[:, 2] >= zmin) & (frac_coords[:, 2] < zmax))[0].tolist()


def remove_random_surface_sites(
        slab, surface_sites_and_indices, remove_ratio, symmetry_frac_tol=1e-5, rng=np.random.default_rng(None),
        n_sample_max=5, remove_atom_types=None,  # Any type can be removed.
        detect_isolated_atom_range=3.0, remove_isolated_atom=True
):
    # Mirror first. If no mirror, then inverse.
    mapping = get_slab_equivalence_groups(
        slab.species, surface_sites_and_indices, frac_tol=symmetry_frac_tol, mode="mirror"
    )
    if mapping is None:
        mapping = get_slab_equivalence_groups(
            slab.species, surface_sites_and_indices, frac_tol=symmetry_frac_tol, mode="inverse"
        )

    if remove_atom_types is None:
        remove_atom_types = [str(site.specie) for site in slab]
    remove_atom_types = set(remove_atom_types)

    vac_slabs = []
    vac_names = []
    top_site_inds = [si[1] for si in surface_sites_and_indices["top"] if str(slab[si[1]].specie) in remove_atom_types]

    if mapping is None:
        # No symmetry applied.
        bot_site_inds = [si[1] for si in surface_sites_and_indices["bottom"] if
                         str(slab[si[1]].specie) in remove_atom_types]

        all_site_inds = top_site_inds + bot_site_inds
        n_surf_sites = len(all_site_inds)
        n_remove = min(int(np.ceil(remove_ratio * n_surf_sites)), n_surf_sites)
        n_sample = min(math.comb(n_surf_sites, n_remove), n_sample_max)
        for samp_id in range(n_sample):
            remove_inds = rng.choice(all_site_inds, n_remove, replace=False).tolist()
            slab_cp = slab.copy()
            slab_cp.remove_sites(remove_inds)
            # Detect and delete isolated atoms.
            if remove_isolated_atom:
                slab_cp = remove_isolated_atoms(slab_cp, radius=detect_isolated_atom_range)
            real_remove_ratio = float(len(slab) - len(slab_cp)) / n_surf_sites
            vac_slabs.append(slab_cp)
            vac_names.append(f"remove_{remove_ratio:.4f}_sample_{samp_id}_actual_{real_remove_ratio:.4f}")
    else:
        # Apply symmetry on top and bottom.
        n_surf_sites = len(top_site_inds)
        n_remove = min(int(np.ceil(remove_ratio * n_surf_sites)), n_surf_sites)
        n_sample = min(math.comb(n_surf_sites, n_remove), n_sample_max)
        for samp_id in range(n_sample):
            remove_inds = rng.choice(top_site_inds, n_remove, replace=False).tolist()
            slab_cp = symmetrically_remove_top_sites(slab, remove_inds, mapping)
            # Detect and delete isolated atoms.
            if remove_isolated_atom:
                slab_cp = remove_isolated_atoms(slab_cp, radius=detect_isolated_atom_range)
            real_remove_ratio = float(len(slab) - len(slab_cp)) / (2 * n_surf_sites)
            vac_slabs.append(slab_cp)
            vac_names.append(f"remove_{remove_ratio:.4f}_sample_{samp_id}_actual_{real_remove_ratio:.4f}")

    return vac_names, vac_slabs, (mapping is None)


def generate_slabs_with_random_vacancies(
        prim, miller_index, symprec=0.1, angle_tol=8, min_slab_ab=12.0, min_slab=12.0, min_vac=20.0,
        max_normal_search=20,
        symmetrize_slab=False, tasker2_modify_polar=True, drop_polar=True,
        remove_atom_types=None,
        min_vacancy_ratio=0.0, max_vacancy_ratio=0.3, num_vacancy_ratios=1, n_sample_per_ratio=5,
        surface_mapping_fractol=1e-5,
        seed=None, detect_isolated_atom_range=3.0, remove_isolated_atom=True,
        max_return_slabs=500,  # If more than this, randomly sample this number. Prevents overly bulky computations.
):
    # Symmetrize makes charge unbalanced slabs in ionic materials. Recommend not to use.
    # Tasker would be helpful, but not always necessary.
    slabs = get_slabs(
        prim, miller_index,
        symprec=symprec, angle_tol=angle_tol,
        min_slab_ab=min_slab_ab, min_slab=min_slab, min_vac=min_vac,
        max_normal_search=max_normal_search,
        symmetrize_slab=symmetrize_slab,
        tasker2_modify_polar=tasker2_modify_polar,
        drop_polar=drop_polar,
    )
    rng = np.random.default_rng(seed)
    remove_ratios = np.linspace(min_vacancy_ratio, max_vacancy_ratio, num_vacancy_ratios)
    vac_slabs = []
    vac_names = []

    if max_vacancy_ratio > 0 and any(get_site_charge(site) != 0 for site in prim):
        print(
            "Warning: Removing atoms from ionic slab. May result in charge imbalance."
            " Make sure you know what you are doing!"
        )

    if max_vacancy_ratio == 0:
        vac_names = [
            f"miller_{"_".join([str(ii) for ii in miller_index])}_slab_{slab_id}_remove_0.0000_sample_0_actual_0.0000"
            for slab_id in range(len(slabs))
        ]
        vac_slabs = slabs
        return vac_names, vac_slabs, slabs

    for slab_id, slab in enumerate(slabs):
        print(f"Generating random vacancies, slab: {slab_id + 1}/{len(slabs)}.")
        surface_sites_and_indices = slab.get_surface_sites()
        prefix = f"miller_{"_".join([str(ii) for ii in miller_index])}_slab_{slab_id}_"
        for remove_ratio in remove_ratios:
            sub_names, sub_slabs, no_symmetry = remove_random_surface_sites(
                slab, surface_sites_and_indices, remove_ratio,
                symmetry_frac_tol=surface_mapping_fractol,
                rng=rng,
                remove_atom_types=remove_atom_types,
                n_sample_max=n_sample_per_ratio,
                detect_isolated_atom_range=detect_isolated_atom_range,
                remove_isolated_atom=remove_isolated_atom,
            )
            sub_names = [prefix + name for name in sub_names]
            vac_slabs.extend(sub_slabs)
            vac_names.extend(sub_names)
        if no_symmetry:
            print(
                "Warning: The two surfaces of the slab are not related by inverse or mirror."
                " Symmetry constraint will be ignored, therefore site removal may result in extra"
                " dipole in slab model with vacancy."
            )

    n_total = len(vac_names)
    if n_total > max_return_slabs:  # Require sub-sampling.
        print(f"Warning: more structures ({n_total}) generated than required ({max_return_slabs}). Down sampling.")
        sample_inds = rng.choice(n_total, size=max_return_slabs, replace=False).astype(int).tolist()
        vac_names = [vac_names[i] for i in range(n_total) if i in sample_inds]
        vac_slabs = [vac_slabs[i] for i in range(n_total) if i in sample_inds]
    # Replace slash with underscore to prevent filename saving issues.
    vac_names = [name.replace("/", "_") for name in vac_names]
    return vac_names, vac_slabs, slabs  # slabs are to be saved into json.