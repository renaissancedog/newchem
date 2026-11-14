"""
Read and inspect an HDF5 (.h5) file.

Usage:
    python read_h5.py path/to/file.h5 [--key KEY]
    python read_h5.py "path/to/shard_glob*.h5" --key KEY --combine [--out combined.h5]
"""

import argparse
import glob
import sys, random, os, json

import h5py
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)


def read_with_pandas(path: str, key: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a specific key from an HDF5 file as a pandas DataFrame."""
    df = pd.read_hdf(path, key=key, columns=columns)
    #df = df[df["Li_first_dE"] <-1.5]
    #df = df[df["status"] == "sodiated"]
    #df = df[np.isnan(df["Li_first_dE"])]
    #df = df[df["formula_pretty"].str.contains("Li")]
    for i, row in df.iterrows():
        if i>2:
            continue
        visualize_row(row)
    #df=df.drop(columns=['mp_structure', 'data', 'host_structure', 'Na_structure', 'Li_structure'])
    print(df.columns)
    print(df.describe())
    print(df.head(30))
    return df

def visualize_row(row, output_dir: str = "visuals") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.visualize import view
    from ase.visualize.plot import plot_atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    import os

    material_id = row["material_id"]
    print(f"Selected row: material_id={material_id}")

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    def save_structure(structure_json: str, label: str):
        if not isinstance(structure_json, str) or not structure_json.strip():
            return
        try:
            structure = Structure.from_str(structure_json, fmt="json")
            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            print(f"  Skipping '{label}': could not parse structure ({type(e).__name__}: {e})")
            return

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{material_id}_{label}.poscar")
        try:
            structure.to(filename=out_path, fmt="poscar")
        except Exception as e:
            print(f"  Failed to write '{label}': {type(e).__name__}: {e}") 

        fig, ax = plt.subplots(figsize=(4, 4))
        plot_atoms(atoms, ax, radii=0.5, rotation="30x,30y,0z")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        out_path = os.path.join(output_dir, f"{material_id}_{label}.png")
        #fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        #saved_paths.append(out_path)
        #print(f"  Saved '{label}' -> {out_path}")

    host_value = row["host_structure"]
    try:
        save_structure(host_value, "data_host")
    except Exception as e:
        print(f"  Could not parse 'data' column: {type(e).__name__}: {e}")

    data_value = row["data"]
    if isinstance(data_value, str) and data_value.strip() not in ("", "[]"):
        try:
            trajectory = json.loads(data_value)
        except Exception as e:
            print(f"  Could not parse 'data' column: {type(e).__name__}: {e}")
            trajectory = []
        for n, structure_json in trajectory:
            structure = Structure.from_str(structure_json, fmt="json")
            atoms = AseAtomsAdaptor.get_atoms(structure)
            #view(atoms)
            save_structure(structure_json, f"data_N{n}")

    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Read an HDF5 (.h5) file.")
    parser.add_argument("path", help="Path to the .h5 file, or a glob pattern when --combine is set")
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
        "--combine",
        action="store_true",
        help="Treat 'path' as a glob pattern matching multiple shard files, "
        "read each with --key/--columns, and concatenate them into one DataFrame.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="With --combine, optionally write the combined DataFrame to this HDF5 path "
        "(uses --key as the write key too). With --reset, the path to write the reset "
        "DataFrame to; defaults to overwriting 'path' in place if omitted.",
    )
    args = parser.parse_args()

    if args.combine:
        if args.key is None:
            print("--combine requires --key", file=sys.stderr)
            sys.exit(1)
        matches = sorted(glob.glob(args.path))
        if not matches:
            print(f"No files matched glob pattern '{args.path}'", file=sys.stderr)
            sys.exit(1)
        print(f"Combining {len(matches)} file(s):")
        frames = []
        for m in matches:
            df = read_with_pandas(m, args.key)
            print(f"  {m}: {len(df)} rows")
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        print(f"\nCombined DataFrame ({len(combined)} rows):")
        print(combined.shape)
        #print(combined)
        if args.out:
            combined.to_hdf(args.out, key=args.key, mode="w")
            print(f"\nWrote combined DataFrame to {args.out} (key: {args.key})")
        return

    if args.key is None:
        print("\nNo --key given, so nothing was loaded into a DataFrame.")
        print("Re-run with --key <one of the names above> to load data, e.g.:")
        print(f"  python read_h5.py {args.path} --key {available_keys[0] if available_keys else '<key>'}")
        return

    try:
        df = read_with_pandas(args.path, args.key)
    except Exception as e:
        print(f"\nCould not read key '{args.key}' with pandas: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
