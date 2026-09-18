# examples/full_mp_interstitial/get_data.py
#
# Full-Materials-Project interstitial intercalation run.
#
# Unlike the insertion-electrode examples, this one starts from every experimentally
# observed near-hull material and discovers its own insertion sites, so it is not limited
# to hosts the Materials Project already reports as electrodes.
#
# Configuration is the block of module-level constants below (no argparse), matching the
# other examples. To use several GPUs, launch one task per GPU with srun: each task shards
# the work automatically from SLURM_PROCID/SLURM_NTASKS and writes its own output file.
import logging
import os
import sys

import pandas as pd

# Ensure the mpelectroml package can be imported
if __name__ == '__main__':
    examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(examples_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from mpelectroml.data_retrieval import get_materials_summary
from mpelectroml.intercalation import IntercalationSettings, add_intercalation_data_to_df
from mpelectroml.utils import get_api_key, setup_logging, HDF5_KEY_INTERCALATION

# --- Configuration ---
FILE_DIRPATH = "."
MATERIALS_HDF5 = "mp_materials.h5"          # cached MP query result
OUTPUT_FILENAME = "Li_Na_intercalation_data.h5"

# Materials Project query. Fields beyond material_id/structure are stored for later
# analysis; energy_per_atom in particular cannot be backfilled without re-querying.
MP_SUMMARY_FIELDS = ["material_id", "structure", "formula_pretty",
                     "energy_per_atom", "energy_above_hull"]
MP_SEARCH_FILTERS = {"theoretical": False, "energy_above_hull": (0, 0.1)}

SKIP_MP_RETRIEVAL = True   # reuse MATERIALS_HDF5 instead of querying MP

# Sharding. Under SLURM each task takes a strided slice of the size-sorted frame
# (rank::ntasks) rather than a contiguous block, so cheap and expensive structures are
# spread evenly and tasks finish at roughly the same time. Each shard writes its own file.
SHARD_RANK = int(os.environ.get("SLURM_PROCID", 0))
SHARD_COUNT = int(os.environ.get("SLURM_NTASKS", 1))

# Continue from an existing output file, skipping rows already completed. Set False to
# recompute a shard from scratch.
RESUME = True

# Optional further bounds applied *within* this shard.
IDX_INIT = 0
IDX_FINAL = -1
CHECKPOINT_EVERY = 1

# Model and relaxation settings for this run. These are deliberately set here rather than
# relying on library defaults, which are tuned for the insertion-electrode pipeline.
SETTINGS = IntercalationSettings(
    working_ion="Li",
    new_working_ion="Na",
    bulk_energies={"Li": -1.9032, "Na": -1.3093},
    min_host_distance={"Li": 1.75, "Na": 2.15},
    merge_distance=0.25,
    symprec=0.1,
    dedup_distance=0.1,
    max_cell_growth=0.15,
    first_ion_energy_cutoff=-1.5,
    fmax=0.02,
    steps=500,
    optimizer="fire",
    optimizer_kwargs={"dt": 0.05, "maxstep": 0.1, "dtmax": 0.2, "downhill_check": False},
    model_name="uma-s-1p2",
    device="cuda",
    task_name="omat",
    cache_dir=os.environ.get("FAIRCHEM_CACHE_DIR"),
)

LOG_LEVEL = "INFO"
LOG_FILE_NAME = "full_mp_interstitial.log"


def run_analysis_workflow():
    """Retrieves candidate materials, then runs the intercalation pipeline over a shard."""
    logger = logging.getLogger(__name__)
    materials_path = os.path.join(FILE_DIRPATH, MATERIALS_HDF5)

    if SKIP_MP_RETRIEVAL or os.path.exists(materials_path):
        logger.info(f"Loading cached materials from {materials_path}")
        df = pd.read_hdf(materials_path)
    else:
        api_key = get_api_key()
        if not api_key:
            logger.error("MP_API_KEY not found. Cannot retrieve materials.")
            return
        df = get_materials_summary(api_key, MP_SUMMARY_FIELDS, **MP_SEARCH_FILTERS)
        if df.empty:
            logger.error("No materials retrieved; nothing to do.")
            return
        os.makedirs(FILE_DIRPATH, exist_ok=True)
        df.to_hdf(materials_path, key=HDF5_KEY_INTERCALATION, mode="w")

    # Cheapest structures first, so that a shard's cost is dominated by the tail rather
    # than by where its boundaries happen to fall.
    df = df.sort_values("num_sites").reset_index(drop=True)
    output_filename = OUTPUT_FILENAME

    random_array = [282, 2407, 2777, 3018, 3227, 4404, 4560, 4923, 5044, 5209, 5727, 5790, 6630, 7206,
                     7432, 7700, 7974, 8252, 8489, 8685, 8795, 8918, 9877, 10003, 10148, 10214, 10872,
                     11285, 11312, 11358, 11787, 12070, 12577, 12950, 14097, 15887, 16252, 16315, 16462,
                     16700, 16914, 18153, 18815, 19880, 20136, 20309, 20396, 20497, 20784, 21185, 23170,
                     23625, 24085, 24509, 24629, 24893, 24931, 25085, 25112, 26416, 26502, 26835, 27565,
                     27642, 28435, 28440, 28461, 28621, 29020, 29913, 29954, 30037, 31104, 31263, 32178,
                     32446, 32624, 32642, 33446, 33562, 33814, 34153, 34265, 34349, 34506, 34972, 35751,
                     35880, 36866, 36982, 37545, 37641, 37672, 37673, 37816, 38181, 38406, 38801, 39467,
                     39669]

    if SHARD_COUNT > 1:
        shard_indices = [idx // SHARD_COUNT for idx in random_array if idx % SHARD_COUNT == SHARD_RANK]
        df = df.iloc[SHARD_RANK::SHARD_COUNT].reset_index(drop=True)
        output_filename = OUTPUT_FILENAME.replace(".h5", f"_shard{SHARD_RANK}.h5")
    else:
        shard_indices = [idx for idx in random_array if idx < len(df)]

    logger.info(f"shard {SHARD_RANK + 1}/{SHARD_COUNT}: {len(df)} materials, "
                f"{len(shard_indices)} target rows -> {output_filename}")

    for idx in shard_indices:
        df = add_intercalation_data_to_df(
            df,
            settings=SETTINGS,
            file_dirpath=FILE_DIRPATH,
            structure_column="mp_structure",
            idx_init=idx,
            idx_final=idx + 1,
            checkpoint_every=CHECKPOINT_EVERY,
            output_filename=output_filename,
            resume=RESUME,
        )
    logger.info("Workflow complete.")

if __name__ == '__main__':
    setup_logging(level=getattr(logging, LOG_LEVEL), log_file=LOG_FILE_NAME)
    run_analysis_workflow()
