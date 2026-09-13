"""
Read and inspect an HDF5 (.h5) file.

Usage:
    python read_h5.py path/to/file.h5 [--key KEY]

If --key is omitted, the script lists all top-level keys/groups in the file
so you can see what's available, then tries to load the data with pandas.
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd

# Force pandas to display everything: no row/column truncation, no line wrapping,
# and no truncation of long cell contents.
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# If any cell contains a NumPy array (common for structure/site data), pandas'
# settings above don't stop *that array's own* repr from truncating - NumPy
# truncates arrays over ~1000 elements by default regardless of pandas config.
np.set_printoptions(threshold=sys.maxsize, linewidth=200)


def list_h5_contents(path: str) -> list[str]:
    """Print the structure of an HDF5 file and return all dataset/group paths."""
    paths = []

    def _visitor(name, obj):
        kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        shape = getattr(obj, "shape", "")
        dtype = getattr(obj, "dtype", "")
        print(f"  {kind:7s} {name}  {shape} {dtype}")
        paths.append(name)

    with h5py.File(path, "r") as f:
        print(f"Contents of {path}:")
        f.visititems(_visitor)

    return paths


def read_with_pandas(path: str, key: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a specific key from an HDF5 file as a pandas DataFrame.

    If `columns` is given, only those columns are loaded. Note: this only
    works if the table was written in 'table' format (pd.to_hdf(..., format='table')).
    Fixed-format ('fixed', the default) HDF5 tables must be loaded in full and
    then sliced afterward.
    """
    random_array = [16034, 12098, 2731, 12013, 9486, 9356, 12309, 14530, 2962, 5886, 1595, 7901, 7548, 14332, 7734]
    df = pd.concat([pd.read_hdf(path, key=key, columns=columns, start=i, stop=i + 1) for i in random_array])
    #df = pd.read_hdf(path, key=key, columns=columns)
    #df = pd.read_hdf(path, key=key, columns=columns)
    return df.drop(columns=['data', 'mp_structure'])
    return df.drop(columns=['data', 'mp_structure', 'Li_structure', 'Na_structure', 'host_structure'])

def main():
    parser = argparse.ArgumentParser(description="Read an HDF5 (.h5) file.")
    parser.add_argument("path", help="Path to the .h5 file")
    parser.add_argument(
        "--key",
        default=None,
        help="HDF5 key/group to load as a pandas DataFrame (e.g. 'intercalation'). "
        "If omitted, only the file structure is listed.",
    )
    parser.add_argument(
       "--list-columns",
        action="store_true",
        help="Just print the column names for --key and exit (no row data).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Only load these specific columns, e.g. --columns col1 col2 col3",
    )
    args = parser.parse_args()

    # Always show the structure first — useful for figuring out valid keys.
    try:
        available_keys = list_h5_contents(args.path)
    except Exception as e:
        print(f"Could not open '{args.path}' as an HDF5 file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.key is None:
        print("\nNo --key given, so nothing was loaded into a DataFrame.")
        print("Re-run with --key <one of the names above> to load data, e.g.:")
        print(f"  python read_h5.py {args.path} --key {available_keys[0] if available_keys else '<key>'}")
        return

    if args.list_columns:
        try:
            cols = pd.read_hdf(args.path, key=args.key, stop=0).columns
        except Exception as e:
            print(f"\nCould not read columns for key '{args.key}': {e}", file=sys.stderr)
            sys.exit(1)
        print(f"\nColumns in key '{args.key}' ({len(cols)} total):")
        for c in cols:
            print(f"  - {c}")
        return

    try:
        df = read_with_pandas(args.path, args.key, columns=args.columns)
    except Exception as e:
        print(f"\nCould not read key '{args.key}' with pandas: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded DataFrame from key '{args.key}':")
    print(df.shape)
    print(df)


if __name__ == "__main__":
    main()
