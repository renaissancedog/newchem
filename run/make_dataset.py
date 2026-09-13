# examples/full_mp_interstitial/make_dataset.py
#
# Turns the intercalation HDF5 produced by get_data.py into train/test/val CSVs.
#
# Replaces the standalone create_csv.py script: the column selection, voltage definition,
# and 80/10/10 split now come from mpelectroml.datasets, so this dataset and the
# insertion-electrode dataset are built by the same code and share one voltage convention.
import glob
import logging
import os
import sys

import pandas as pd

if __name__ == '__main__':
    examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(examples_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from mpelectroml.datasets import compute_voltages, normalize_dataset, split_and_export
from mpelectroml.utils import setup_logging

# Shards written by get_data.py; a glob so multi-GPU runs concatenate transparently.
INPUT_GLOB = "Li_Na_intercalation_data*.h5"
OUT_DIRPATH = "li_data_interstitial"
SOURCE_NAME = "full_mp_interstitial"

# Only fully processed hosts carry both a Li and a Na energy.
KEEP_STATUS = "sodiated"

# Inclusive, matching the published dataset's "<= 20 atoms/cell" convention.
MAX_ATOMS = 20


def main():
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        raise FileNotFoundError(f"No input files matching {INPUT_GLOB!r}. Run get_data.py first.")

    df = pd.concat([pd.read_hdf(path) for path in paths], ignore_index=True)
    logging.info(f"Loaded {len(df)} rows from {len(paths)} file(s).")

    df = df[df["status"] == KEEP_STATUS].reset_index(drop=True)
    logging.info(f"{len(df)} rows with status {KEEP_STATUS!r}.")

    df = normalize_dataset(df, source=SOURCE_NAME, working_ion="Li", new_working_ion="Na")
    df = compute_voltages(df, working_ions=("Li", "Na"))

    counts = split_and_export(df, OUT_DIRPATH, max_atoms=MAX_ATOMS)
    logging.info(f"Wrote {counts} to {OUT_DIRPATH}/")


if __name__ == '__main__':
    setup_logging()
    main()
