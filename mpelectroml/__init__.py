# mpelectroml/__init__.py
"""
MPElectroML: Materials Project Electrodes with Machine-Learned Interatomic Potentials.

This package provides tools to retrieve electrode material data from the Materials Project,
perform structural modifications (e.g., substituting working ions), and calculate
energies and forces using machine-learned interatomic potentials via FAIRChem.
"""
__version__ = "0.1.0"

# Expose key functions from submodules for easier access
from .data_retrieval import (
    get_electrode_pairs,
    get_structures_from_electrode_pair_ids,
    get_materials_summary
)
from .structure_manipulation import (
    create_new_working_ion_discharge_structures,
    generate_multiples,
    get_interstitial_sites,
    framework_bonds_changed,
    cell_growth_exceeded,
    get_inserted_ion_indices
)
from .calculations import (
    add_energy_forces_to_df,
    calculate_energy_and_forces_from_Structure,
    assign_calculator,
    relax_atoms,
    relax_structure
)
from .intercalation import IntercalationSettings, process_structure, add_intercalation_data_to_df
from .datasets import normalize_dataset, compute_voltages, merge_datasets, split_and_export
from .utils import (
    get_api_key,
    setup_logging,
    HDF5_KEY_ELECTRODE_PAIRS,
    HDF5_KEY_WITH_ENERGIES,
    HDF5_KEY_INTERCALATION
)

__all__ = [
    "get_electrode_pairs", "get_structures_from_electrode_pair_ids", "get_materials_summary",
    "create_new_working_ion_discharge_structures", "generate_multiples",
    "get_interstitial_sites", "framework_bonds_changed", "cell_growth_exceeded",
    "get_inserted_ion_indices",
    "add_energy_forces_to_df", "calculate_energy_and_forces_from_Structure",
    "assign_calculator", "relax_atoms", "relax_structure",
    "IntercalationSettings", "process_structure", "add_intercalation_data_to_df",
    "normalize_dataset", "compute_voltages", "merge_datasets", "split_and_export",
    "get_api_key", "setup_logging",
    "HDF5_KEY_ELECTRODE_PAIRS", "HDF5_KEY_WITH_ENERGIES", "HDF5_KEY_INTERCALATION"
]
