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
    fmax=0.1,
    steps=500,
    optimizer="fire",
    optimizer_kwargs={"dt": 0.05, "maxstep": 0.1, "dtmax": 0.2, "downhill_check": False},
    model_name="uma-s-1p2p1",
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

    random_array=[64, 73, 130, 192, 212, 342, 345, 348, 390, 549, 644, 740, 850, 966, 1130, 1144, 1203, 1220, 1222, 1252, 1297, 1391, 1469, 1597, 1623, 1785, 1835, 1847, 1902, 1908, 1942, 2078, 2116, 2200, 2329, 2498, 2639, 2713, 2725, 2755, 2876, 2920, 3003, 3303, 3305, 3382, 3499, 3533, 3647, 3690, 3722, 3858, 3954, 3963, 3975, 4011, 4295, 4318, 4366, 4379, 4411, 4589, 4713, 4772, 4826, 4877, 5099, 5109, 5177, 5296, 5457, 5509, 5548, 5569, 5682, 5959, 6010, 6025, 6075, 6135, 6150, 6263, 6351, 6353, 6396, 6492, 6527, 6654, 6710, 6723, 6724, 6777, 6989, 7050, 7086, 7235, 7239, 7291, 7408, 7484, 7590, 7660, 7662, 7722, 7755, 7777, 7899, 7919, 8029, 8065, 8103, 8126, 8239, 8343, 8423, 8428, 8431, 8452, 8545, 8568, 8589, 8866, 8887, 8930, 9003, 9025, 9191, 9238, 9334, 9347, 9608, 9785, 9819, 9954, 9969, 10012, 10049, 10062, 10118, 10308, 10595, 10683, 10715, 10948, 11048, 11128, 11143, 11162, 11300, 11386, 11420, 11545, 11579, 11635, 11704, 11770, 11874, 11970, 12070, 12153, 12254, 12269, 12339, 12408, 12409, 12464, 12500, 12559, 12666, 12690, 12696, 12850, 12912, 12957, 13022, 13034, 13102, 13107, 13335, 13459, 13468, 13487, 13555, 13595, 13617, 13671, 13672, 13720, 13759, 13916, 14068, 14220, 14257, 14321, 14365, 14479, 14492, 14505, 14554, 14743, 14793, 14795, 15088, 15183, 15191, 15290, 15376, 15589, 15701, 15703, 15729, 15759, 15827, 15903, 15940, 16152, 16252, 16426, 16476, 16636, 16694, 16840, 16910, 16922, 16931, 16993, 17088, 17125, 17146, 17213, 17228, 17270, 17324, 17339, 17458, 17531, 17676, 17752, 17762, 17844, 17873, 17910, 17911, 17916, 18073, 18084, 18199, 18231, 18250, 18494, 18495, 18529, 18625, 18626, 18672, 18708, 18790, 18907, 18914, 18965, 18990, 19191, 19194, 19465, 19470, 19772, 19777, 19800, 19850, 20155, 20195, 20243, 20313, 20315, 20343, 20361, 20374, 20438, 20444, 20496, 20521, 20660, 20675, 20805, 20972, 21216, 21227, 21242, 21264, 21383, 21814, 21853, 21856, 21875, 21902, 21917, 22013, 22043, 22085, 22103, 22282, 22389, 22610, 22752, 23124, 23135, 23138, 23141, 23143, 23174, 23178, 23196, 23382, 23390, 23479, 23540, 23557, 23579, 23583, 23614, 23836, 23847, 23904, 23986, 24011, 24023, 24110, 24140, 24186, 24190, 24323, 24438, 24513, 24562, 24930, 24997, 25009, 25019, 25195, 25416, 25429, 25485, 25512, 25660, 25685, 25845, 26117, 26136, 26238, 26253, 26390, 26423, 26433, 26451, 26565, 26646, 26659, 26712, 26785, 26841, 26910, 26968, 27067, 27361, 27466, 27490, 27513, 27581, 27592, 27617, 27620, 27642, 27721, 27965, 28099, 28158, 28210, 28263, 28288, 28561, 28746, 28908, 29156, 29166, 29360, 29382, 29490, 29593, 29719, 29762, 29776, 29895, 29974, 29983, 30078, 30199, 30210, 30252, 30267, 30659, 30688, 30728, 30737, 31186, 31421, 31566, 32543, 32688, 32743, 32841, 32853, 32866, 32943, 33043, 33098, 33167, 33179, 33201, 33305, 33324, 33467, 33522, 33670, 33722, 33962, 34030, 34195, 34196, 34260, 34310, 34345, 34406, 34514, 34517, 34707, 34736, 34822, 34826, 34835, 34859, 34905, 35084, 35172, 35191, 35342, 35374, 35473, 35493, 35571, 35712, 35767, 35802, 35879, 36002, 36082, 36134, 36282, 36455, 36477, 36570, 36734, 36756, 36775, 36787, 36939, 36954, 37218, 37226, 37341, 37409, 37555, 37626, 37726, 37758, 37769, 37791, 37959, 37985, 38035, 38054, 38130, 38285, 38313, 38339, 38560, 38576, 38661, 38853, 38879, 39073, 39443, 39513, 39728, 39811, 39867, 39895, 39896, 39900, 39965, 40010]
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
