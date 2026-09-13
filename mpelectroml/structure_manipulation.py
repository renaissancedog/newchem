import numpy as np
import pandas as pd
from ase.data import covalent_radii
from scipy.spatial import Voronoi
from pymatgen.core import Structure, Element, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import logging

logger = logging.getLogger(__name__)


def generate_multiples(n: int) -> list[list[int]]:
    """
    Generates all ordered combinations of three positive integers (a, b, c)
    such that a * b * c = n. This is used for determining possible supercell
    transformations.

    Args:
        n (int): The integer for which to find multiplicative factors.
                 Must be a positive integer.

    Returns:
        list[list[int]]: A list of [a, b, c] triplets.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input n must be a positive integer for generate_multiples.")

    results = []
    for a in range(1, n + 1):
        if n % a == 0:
            remaining_n_div_a = n // a
            for b in range(1, remaining_n_div_a + 1):
                if remaining_n_div_a % b == 0:
                    c = remaining_n_div_a // b
                    results.append([a, b, c])
    return results


def create_new_working_ion_discharge_structures(
    df_pairs: pd.DataFrame,
    original_working_ion: str = "Li",
    new_working_ion: str = "Na"
) -> None:
    """
    Creates new discharge Pymatgen Structure objects by replacing the `original_working_ion`
    with the `new_working_ion` in the 'discharge_structure' column.
    This function attempts to handle cases where the host frameworks (after removing
    working ions) of the charge and discharge structures do not initially match in size,
    by applying supercell transformations.

    Modifies the input DataFrame `df_pairs` in place by adding columns:
    - f"{new_working_ion}_discharge_structure" (Pymatgen Structure objects or None)
    - f"{new_working_ion}_discharge_formula" (string chemical formulas or None)

    Args:
        df_pairs (pd.DataFrame): DataFrame which must contain 'charge_structure' and
                                 'discharge_structure' columns with Pymatgen Structure objects.
        original_working_ion (str): The chemical symbol of the working ion to be replaced (e.g., "Li").
        new_working_ion (str): The chemical symbol of the new working ion to substitute (e.g., "Na").
    """
    new_struct_col_name = f"{new_working_ion}_discharge_structure"
    new_formula_col_name = f"{new_working_ion}_discharge_formula"

    # Initialize new columns with None if they don't exist
    if new_struct_col_name not in df_pairs.columns:
        df_pairs[new_struct_col_name] = pd.Series([None] * len(df_pairs), dtype=object)
    if new_formula_col_name not in df_pairs.columns:
        df_pairs[new_formula_col_name] = pd.Series([None] * len(df_pairs), dtype=object)

    new_discharge_structure_list = []
    match_stats = {"unmatched_complex": 0, "original_ion_in_charge_host": 0, "processed_rows": 0}

    for i in range(df_pairs.shape[0]):
        original_working_ion_i = (str(df_pairs.at[i, 'working_ion'])
                                  if original_working_ion == '' else original_working_ion)
        match_stats["processed_rows"] += 1
        charge_struct_pmg = df_pairs.at[i, 'charge_structure']
        discharge_struct_pmg = df_pairs.at[i, 'discharge_structure']

        # Ensure we have Pymatgen Structure objects to work with
        if not isinstance(charge_struct_pmg, Structure) or not isinstance(discharge_struct_pmg, Structure):
            logger.debug(f"Row {i}: Skipping due to missing Pymatgen charge or discharge structure.")
            new_discharge_structure_list.append(None)
            continue

        charge_struct_pmg_copy = charge_struct_pmg.copy()
        discharge_struct_pmg_copy = discharge_struct_pmg.copy()
        new_discharge_structure = discharge_struct_pmg.copy()

        if original_working_ion_i in Composition(charge_struct_pmg.formula).get_el_amt_dict():
            match_stats["original_ion_in_charge_host"] += 1

            # Create host frameworks by removing the original working ion
            charge_host = charge_struct_pmg_copy.copy()
            charge_host.remove_species([original_working_ion_i])

            discharge_host = discharge_struct_pmg_copy.copy()
            discharge_host.remove_species([original_working_ion_i])

            if charge_host.composition.formula != discharge_host.composition.formula:
                logger.debug(f"Row {i}: Host formulas differ for {original_working_ion_i} removal. "
                             f"Charge host: {charge_host.composition.formula}, "
                             f"Discharge host: {discharge_host.composition.formula}. Attempting supercell matching.")

                len_charge_host = len(charge_host)
                len_discharge_host = len(discharge_host)

                scaling_ratio = 0
                smaller_structure = None
                larger_structure = None

                if len_discharge_host % len_charge_host == 0:
                    scaling_ratio = round(len_discharge_host / len_charge_host)
                    smaller_structure = charge_host
                    larger_structure = discharge_host
                elif len_charge_host % len_discharge_host == 0:
                    scaling_ratio = round(len_charge_host / len_discharge_host)
                    smaller_structure = discharge_host
                    larger_structure = charge_host

                if scaling_ratio > 0 and smaller_structure and larger_structure:
                    na_nb_nc_options = generate_multiples(scaling_ratio)

                    smaller_lattice = smaller_structure.lattice
                    larger_lattice = larger_structure.lattice

                    na = larger_lattice.a / smaller_lattice.a
                    nb = larger_lattice.b / smaller_lattice.b
                    nc = larger_lattice.c / smaller_lattice.c

                    closest_index = np.argmin(np.linalg.norm(np.array(na_nb_nc_options) -
                                                             np.array([na, nb, nc]), axis=1))
                    na, nb, nc = na_nb_nc_options[closest_index]

                    if len_charge_host < len_discharge_host:
                        charge_struct_pmg_copy = charge_struct_pmg_copy.make_supercell([na, nb, nc])
                    else:
                        discharge_struct_pmg_copy = discharge_struct_pmg_copy.make_supercell([na, nb, nc])

                else:
                    logger.warning(f"Row {i}: Atom number ratio for hosts is not an integer multiple or one host is "
                                   "empty. Cannot scale for matching.")
                    new_discharge_structure_list.append(None)
                    continue

            matcher = StructureMatcher(ltol=0.6, stol=0.8, angle_tol=20, primitive_cell=False,
                                       scale=False, allow_subset=True)
            mapping = matcher.get_mapping(discharge_struct_pmg_copy, charge_struct_pmg_copy)

            new_discharge_structure = discharge_struct_pmg_copy.copy()

            if mapping is None:
                logger.warning(f"Row {i}: Structures do not match after scaling attempt for {original_working_ion_i}. "
                               "No replacement will be made.")
                match_stats["unmatched_complex"] += 1
                new_discharge_structure = None
            else:
                for j, site in enumerate(new_discharge_structure):
                    if j not in mapping:
                        if Element(original_working_ion_i) in site:
                            new_discharge_structure.replace(j, new_working_ion)
                        else:
                            logger.warning(f"Row {i}: Site {j} in discharge structure does not match any site in "
                                           f"charge structure and it is not {original_working_ion_i} but "
                                           f"{site.species_string}.")
        else:
            logger.debug(f"Row {i}: '{original_working_ion_i}' not in charge formula. "
                         f"Attempting simple replacement in discharge structure "
                         f"for '{new_working_ion}'.")
            if original_working_ion_i in Composition(new_discharge_structure.formula).get_el_amt_dict():
                new_discharge_structure.replace_species({original_working_ion_i: new_working_ion})
                new_discharge_structure = new_discharge_structure
            else:
                logger.warning(f"Row {i}: Simple replacement: '{original_working_ion_i}' "
                               f"not found in discharge formula "
                               f"'{new_discharge_structure.formula}'. Cannot create "
                               f"'{new_working_ion}' variant. Structure set to None.")
                new_discharge_structure = None

        new_discharge_structure_list.append(new_discharge_structure)

    df_pairs[new_struct_col_name] = new_discharge_structure_list
    df_pairs[new_formula_col_name] = [s.composition.formula if isinstance(s, Structure) else None
                                      for s in new_discharge_structure_list]

    logger.info(f"Finished creating new discharge structures for ion: '{new_working_ion}'.")
    logger.info(f"Stats for '{new_working_ion}': "
                f"Total rows processed: {match_stats['processed_rows']}. Rows where original ion "
                f"('{original_working_ion}') was in charge host: {match_stats['original_ion_in_charge_host']}. "
                f"Rows where complex matching failed to replace ions: {match_stats['unmatched_complex']}. "
                f"Successfully created structures: {df_pairs[new_struct_col_name].notna().sum()}.")


# ---------------------------------------------------------------------------
# Interstitial site discovery and structural-integrity checks.
#
# These support the "full-MP" pipeline, which inserts working ions into voids
# discovered in an arbitrary host rather than onto sites the Materials Project
# already reports (which is what create_new_working_ion_discharge_structures
# above does for known insertion-electrode pairs).
# ---------------------------------------------------------------------------

# Minimum allowed distance (Angstrom) between an inserted ion and any host atom.
MIN_ION_HOST_DISTANCE = {"Li": 1.75, "Na": 2.15}
DEFAULT_MIN_ION_HOST_DISTANCE = 1.75


def _voronoi_void_sites(structure: Structure, min_host_distance: float) -> list:
    """
    Finds candidate interstitial voids as Voronoi vertices that are clear of host atoms.

    A 3x3x3 supercell is used so that vertices near the cell boundary are computed with
    the correct periodic neighbours; only vertices falling inside the central cell are kept.

    Args:
        structure (Structure): The host structure.
        min_host_distance (float): Vertices closer than this to any host atom are discarded.

    Returns:
        list: Fractional coordinates of collision-free voids, largest void first.
    """
    supercell = structure * (3, 3, 3)
    voronoi = Voronoi(supercell.cart_coords)

    # Vertices are in supercell Cartesian space; expressed in the *original* lattice's
    # fractional coordinates the central cell spans [1, 2) along each axis.
    frac_coords = structure.lattice.get_fractional_coords(voronoi.vertices)
    inside = ((frac_coords >= 1.0 - 1e-5) & (frac_coords < 2.0 - 1e-5)).all(axis=1)
    valid_fracs = frac_coords[inside] - 1.0
    valid_carts = structure.lattice.get_cartesian_coords(valid_fracs)

    voids = []
    for frac, cart in zip(valid_fracs, valid_carts):
        if structure.get_sites_in_sphere(cart, min_host_distance):
            continue  # too close to a host atom
        neighbours = structure.get_sites_in_sphere(cart, 10.0)
        nearest = min((distance for _, distance, *_ in neighbours), default=0.0)
        voids.append((frac, nearest))

    # Largest void (farthest from any host atom) first.
    voids.sort(key=lambda void: void[1], reverse=True)
    return [frac for frac, _ in voids]


def _merge_close_sites(structure: Structure, sites: list, merge_distance: float) -> list:
    """Greedily drops sites closer than `merge_distance` to an already-kept site."""
    kept = []
    for frac in sites:
        if kept:
            distances = structure.lattice.get_all_distances([frac], kept)[0]
            if any(distance < merge_distance for distance in distances):
                continue
        kept.append(frac)
    return kept


def get_symmetry_unique_site_indices(structure: Structure, sites: list, symprec: float = 0.1,
                                     dedup_distance: float = 0.1) -> list:
    """
    Indices of one representative per symmetry-equivalent class among `sites`.

    Two sites are equivalent when some symmetry operation of `structure` maps one onto the
    other. Callers that must relax a structure per candidate site can use this to relax one
    site per class instead of all of them, since equivalent sites give the same energy.

    Args:
        structure (Structure): Structure whose space group defines equivalence.
        sites (list): Fractional coordinates of candidate sites.
        symprec (float): Symmetry tolerance for SpacegroupAnalyzer.
        dedup_distance (float): Two sites within this distance are considered identical.

    Returns:
        list: Indices into `sites`, one per equivalence class, in input order.
    """
    symmops = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations()
    lattice = structure.lattice

    representatives, kept = [], []
    for index, frac in enumerate(sites):
        # Generate this site's full symmetry orbit; if any image coincides with an
        # already-kept site, this site is a duplicate.
        orbit = [op.operate(frac) % 1.0 for op in symmops]
        duplicate = any(
            lattice.get_all_distances([image], [other])[0][0] < dedup_distance
            for image in orbit
            for other in kept
        )
        if not duplicate:
            representatives.append(index)
            kept.append(frac)
    return representatives

def _symmetry_unique_sites(structure: Structure, sites: list, symprec: float,
                           dedup_distance: float) -> list:
    """Collapses sites that are equivalent under the structure's space group."""
    try:
        indices = get_symmetry_unique_site_indices(structure, sites, symprec, dedup_distance)
    except Exception as e:
        logger.info(f"Symmetry reduction unavailable ({type(e).__name__}: {e}); using all sites.")
        return sites
    return [sites[index] for index in indices]

def get_interstitial_sites(
    structure: Structure,
    working_ion: str,
    min_host_distance: float | None = None,
    merge_distance: float = 0.25,
    symprec: float = 0.1,
    dedup_distance: float = 0.1
) -> list:
    """
    Finds symmetry-unique candidate interstitial sites for `working_ion` in `structure`.

    Runs three stages: Voronoi void discovery, merging of near-coincident voids, and
    symmetry deduplication.

    Args:
        structure (Structure): The host structure to insert into.
        working_ion (str): Symbol of the ion to be inserted (e.g. "Li").
        min_host_distance (float | None): Minimum ion-host distance. Defaults to the
            per-ion value in MIN_ION_HOST_DISTANCE, else DEFAULT_MIN_ION_HOST_DISTANCE.
        merge_distance (float): Sites closer together than this are merged.
        symprec (float): Symmetry tolerance for SpacegroupAnalyzer.
        dedup_distance (float): Two sites within this distance are considered identical.

    Returns:
        list: Fractional coordinates of candidate sites, largest void first.
    """
    if min_host_distance is None:
        min_host_distance = MIN_ION_HOST_DISTANCE.get(working_ion, DEFAULT_MIN_ION_HOST_DISTANCE)

    sites = _voronoi_void_sites(structure, min_host_distance)
    sites = _merge_close_sites(structure, sites, merge_distance)
    if not sites:
        # Return before touching SpacegroupAnalyzer: nothing to deduplicate, and this
        # avoids symmetry analysis on structures that have no candidate sites at all.
        return []
    return _symmetry_unique_sites(structure, sites, symprec, dedup_distance)

def _framework_bonds(structure: Structure, working_ion: str) -> set:
    """Set of bonded framework (non-working-ion) index pairs, by covalent-radius cutoff."""
    framework = [i for i, site in enumerate(structure) if site.specie.symbol != working_ion]
    radii = np.array([covalent_radii[structure[i].specie.Z] for i in framework])
    cutoffs = 1.5 * (radii[:, None] + radii[None, :])
    fracs = [structure[i].frac_coords for i in framework]
    distances = structure.lattice.get_all_distances(fracs, fracs)
    i_idx, j_idx = np.triu_indices(len(framework), k=1)
    bonded = distances[i_idx, j_idx] <= cutoffs[i_idx, j_idx]
    return set(zip(i_idx[bonded].tolist(), j_idx[bonded].tolist()))


def framework_bonds_changed(structure_a: Structure, structure_b: Structure,
                            working_ion: str) -> bool:
    """
    True if the framework (non-working-ion) bond network differs between two structures.

    Both structures must list their framework atoms in the same order, which holds when
    one is a relaxed version of the other, or an ion-inserted version of it.

    Args:
        structure_a (Structure): Reference structure.
        structure_b (Structure): Structure to compare against the reference.
        working_ion (str): Ion symbol to exclude from the bond network.

    Returns:
        bool: True if the bond networks differ (i.e. the framework topology changed).
    """
    return _framework_bonds(structure_a, working_ion) != _framework_bonds(structure_b, working_ion)


def cell_growth_exceeded(structure: Structure, reference_structure: Structure,
                         max_growth: float = 0.15) -> bool:
    """
    True if any cell vector of `structure` grew by more than `max_growth` (a fraction)
    relative to `reference_structure`. Shrinkage is not flagged.
    """
    reference_abc = np.array(reference_structure.lattice.abc)
    growth = (np.array(structure.lattice.abc) - reference_abc) / reference_abc
    return bool(np.any(growth > max_growth))


def get_inserted_ion_indices(full_structure: Structure, host_structure: Structure,
                             working_ion: str) -> list:
    """
    Indices of working ions in `full_structure` that were inserted, i.e. that do not
    correspond to an ion already present in `host_structure`.

    Host ions are paired to full-structure ions by globally greedy nearest matching
    (closest pair first); whatever remains unclaimed was inserted.

    Args:
        full_structure (Structure): The ion-inserted structure.
        host_structure (Structure): The host it was built from.
        working_ion (str): Ion symbol.

    Returns:
        list: Indices into `full_structure` of the inserted ions.
    """
    full_indices = [i for i, site in enumerate(full_structure) if site.specie.symbol == working_ion]
    host_fracs = [site.frac_coords for site in host_structure if site.specie.symbol == working_ion]

    n_native = len(host_fracs)
    if n_native == 0:
        return full_indices          # bare framework host: every ion is inserted
    if n_native >= len(full_indices):
        return []                    # nothing beyond the host's own ions

    full_fracs = [full_structure[i].frac_coords for i in full_indices]
    distances = full_structure.lattice.get_all_distances(host_fracs, full_fracs)

    pairs = sorted(
        (distances[host][full], host, full)
        for host in range(n_native)
        for full in range(len(full_indices))
    )
    matched_host, claimed_full = set(), set()
    for _, host, full in pairs:
        if host in matched_host or full in claimed_full:
            continue
        matched_host.add(host)
        claimed_full.add(full)
        if len(matched_host) == n_native:
            break

    return [full_indices[i] for i in range(len(full_indices)) if i not in claimed_full]
