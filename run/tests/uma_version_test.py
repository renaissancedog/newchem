"""
Compare relaxed energies from two UMA models on materials picked the same way get_data.py
picks them.

Modeled off intercalation.py: each structure is relaxed with the pipeline's own `_relax`
helper and `IntercalationSettings`, changing only `model_name`, so the optimizer, steps,
fmax, device and task_name are identical between the two models.

TARGET_INDICES, the sort, and SETTINGS below are copies of what get_data.py uses. If you
change them in get_data.py, change them here too, or index N will stop meaning the same
material in both places. Indices are positions in the full sorted frame (before any
sharding), which is also what get_data.py's shard maths resolves to.

Usage:
    python compare_uma_models.py --n 3                      # first 3 entries of TARGET_INDICES
    python compare_uma_models.py --n 5 --random --seed 0    # 5 random entries
    python compare_uma_models.py --indices 64 73 130        # explicit sorted-frame positions
    python compare_uma_models.py --materials mp-1062030     # explicit material ids
    python compare_uma_models.py --n 3 --fmax 0.02          # override a setting

Run it from the directory get_data.py runs in (so mp_materials.h5 resolves the same way), in
an environment where `mpelectroml` is importable.
"""
import argparse
import logging
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from mpelectroml.intercalation import IntercalationSettings, _relax

logger = logging.getLogger(__name__)

MATERIALS_HDF5 = "mp_materials.h5"
TARGET_INDICES = [64, 73, 130, 192, 212, 342, 345, 348, 390, 549, 644, 740, 850, 966, 1130, 1144, 1203, 1220, 1222, 1252, 1297, 1391, 1469, 1597, 1623, 1785, 1835, 1847, 1902, 1908, 1942, 2078, 2116, 2200, 2329, 2498, 2639, 2713, 2725, 2755, 2876, 2920, 3003, 3303, 3305, 3382, 3499, 3533, 3647, 3690, 3722, 3858, 3954, 3963, 3975, 4011, 4295, 4318, 4366, 4379, 4411, 4589, 4713, 4772, 4826, 4877, 5099, 5109, 5177, 5296, 5457, 5509, 5548, 5569, 5682, 5959, 6010, 6025, 6075, 6135, 6150, 6263, 6351, 6353, 6396, 6492, 6527, 6654, 6710, 6723, 6724, 6777, 6989, 7050, 7086, 7235, 7239, 7291, 7408, 7484]
SORT_BY = ["num_sites", "material_id"]
SORT_KIND = "stable"
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
    model_name="uma-s-1p2",   # replaced per model below
    device="cuda",
    task_name="omat",
    cache_dir=os.environ.get("FAIRCHEM_CACHE_DIR"),
)


def load_sorted_frame(materials_path: str) -> pd.DataFrame:
    """Loads the materials frame and sorts it exactly as get_data.py does."""
    df = pd.read_hdf(materials_path)
    return df.sort_values(SORT_BY, kind=SORT_KIND).reset_index(drop=True)


def _to_structure(value) -> Structure:
    return value if isinstance(value, Structure) else Structure.from_str(value, fmt="json")


def select_materials(df: pd.DataFrame, args) -> list[tuple]:
    """Returns [(sorted_index, material_id, Structure)] for the requested materials."""
    column = args.structure_column
    for needed in ("material_id", column):
        if needed not in df.columns:
            sys.exit(f"Error: materials frame has no '{needed}' column.")

    if args.materials:
        positions = []
        for material_id in args.materials:
            hits = df.index[df["material_id"] == material_id]
            if len(hits) == 0:
                print(f"Warning: {material_id} not found in the materials frame, skipping.")
            else:
                positions.append(int(hits[0]))
        source = "--materials"
    elif args.indices:
        positions, source = list(args.indices), "--indices"
    else:
        if args.random:
            positions = random.Random(args.seed).sample(TARGET_INDICES, min(args.n, len(TARGET_INDICES)))
        else:
            positions = TARGET_INDICES[:args.n]
        source = (f"TARGET_INDICES ({'random' if args.random else 'first'} "
                  f"{len(positions)} of {len(TARGET_INDICES)})")

    selected = []
    for position in positions:
        if not 0 <= position < len(df):
            print(f"Warning: index {position} is outside the sorted frame (0..{len(df) - 1}), skipping.")
            continue
        row = df.iloc[position]
        selected.append((position, row["material_id"], _to_structure(row[column])))

    print(f"Selected {len(selected)} material(s) via {source}:")
    for position, material_id, structure in selected:
        print(f"  idx {position:>6}  {material_id:<12} {structure.composition.reduced_formula} "
              f"({len(structure)} atoms)")
    return selected


def relax_with_model(structure: Structure, base_settings: IntercalationSettings,
                     model_name: str) -> dict:
    """Relaxes a copy of `structure` with `model_name`; never raises, errors land in the result."""
    settings = replace(base_settings, model_name=model_name)
    start = time.time()
    try:
        converged, energy, relaxed = _relax(structure.copy(), settings)
    except Exception as e:
        logger.error(f"{model_name} failed: {type(e).__name__}: {e}")
        return {"converged": False, "energy": np.nan, "energy_per_atom": np.nan,
                "volume_per_atom": np.nan, "seconds": time.time() - start,
                "error": f"{type(e).__name__}: {e}"}

    energy = float(energy) if energy is not None else np.nan
    return {
        "converged": bool(converged),
        "energy": energy,
        "energy_per_atom": energy / len(structure),
        "volume_per_atom": (relaxed.volume / len(relaxed)) if relaxed is not None else np.nan,
        "seconds": time.time() - start,
        "error": "",
    }


def compare(selected: list[tuple], base_settings: IntercalationSettings,
            model_a: str, model_b: str) -> pd.DataFrame:
    rows = []
    for position, material_id, structure in selected:
        print(f"\nidx {position} / {material_id} ({structure.composition.reduced_formula}, "
              f"{len(structure)} atoms)")
        results = {}
        for tag, model in (("a", model_a), ("b", model_b)):
            print(f"  relaxing with {model} ...")
            r = results[tag] = relax_with_model(structure, base_settings, model)
            status = "converged" if r["converged"] else "NOT converged"
            print(f"    {status}, E/atom = {r['energy_per_atom']:.6f} eV, {r['seconds']:.1f}s"
                  + (f"  [{r['error']}]" if r["error"] else ""))

        a, b = results["a"], results["b"]
        rows.append({
            "sorted_idx": position,
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "n_atoms": len(structure),
            f"E_per_atom[{model_a}]": a["energy_per_atom"],
            f"E_per_atom[{model_b}]": b["energy_per_atom"],
            "dE_per_atom(b-a)": b["energy_per_atom"] - a["energy_per_atom"],
            "dE_total(b-a)": b["energy"] - a["energy"],
            f"vol_per_atom[{model_a}]": a["volume_per_atom"],
            f"vol_per_atom[{model_b}]": b["volume_per_atom"],
            "dVol_pct(b-a)": 100 * (b["volume_per_atom"] - a["volume_per_atom"]) / a["volume_per_atom"],
            f"converged[{model_a}]": a["converged"],
            f"converged[{model_b}]": b["converged"],
            f"sec[{model_a}]": a["seconds"],
            f"sec[{model_b}]": b["seconds"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--materials-file", default=MATERIALS_HDF5,
                        help=f"Materials HDF5 file (default: {MATERIALS_HDF5})")
    parser.add_argument("--structure-column", default="mp_structure")

    pick = parser.add_argument_group("which materials (default: first --n of TARGET_INDICES)")
    pick.add_argument("--n", type=int, default=100,
                      help="How many entries to take from TARGET_INDICES (default 3)")
    pick.add_argument("--random", action="store_true", default=True,
                      help="Sample --n entries at random instead of taking the first --n")
    pick.add_argument("--seed", type=int, default=1, help="Seed for --random")
    pick.add_argument("--indices", nargs="+", type=int,
                      help="Explicit positions in the sorted frame (overrides TARGET_INDICES)")
    pick.add_argument("--materials", nargs="+",
                      help="Explicit material_ids (overrides --indices and TARGET_INDICES)")

    parser.add_argument("--models", nargs=2, default=["uma-s-1p2", "uma-s-1p2p1"],
                        metavar=("MODEL_A", "MODEL_B"))

    over = parser.add_argument_group("overrides for SETTINGS")
    over.add_argument("--fmax", type=float, default=None)
    over.add_argument("--steps", type=int, default=None)
    over.add_argument("--device", default=None)
    over.add_argument("--task-name", default=None)
    over.add_argument("--cache-dir", default=None)

    parser.add_argument("--out", default="tests/uma_model_comparison.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    overrides = {"fmax": args.fmax, "steps": args.steps, "device": args.device,
                 "task_name": args.task_name, "cache_dir": args.cache_dir}
    base_settings = replace(SETTINGS, **{k: v for k, v in overrides.items() if v is not None})
    print(f"Relaxation settings: fmax={base_settings.fmax}, steps={base_settings.steps}, "
          f"optimizer={base_settings.optimizer}, device={base_settings.device}, "
          f"task_name={base_settings.task_name}")

    if not os.path.exists(args.materials_file):
        sys.exit(f"Error: {args.materials_file} not found. Run from get_data.py's directory "
                 "or pass --materials-file.")
    df = load_sorted_frame(args.materials_file)
    print(f"Loaded {len(df)} materials from {args.materials_file}, "
          f"sorted by {SORT_BY} (kind={SORT_KIND})")

    selected = select_materials(df, args)
    if not selected:
        sys.exit("No materials selected; nothing to do.")

    model_a, model_b = args.models
    table = compare(selected, base_settings, model_a, model_b)

    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    print(f"\n=== Comparison (b - a means {model_b} minus {model_a}) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    ok = table[f"converged[{model_a}]"] & table[f"converged[{model_b}]"]
    both = table[ok]
    print(f"\n{len(both)}/{len(table)} material(s) converged under both models.")
    if len(both):
        diffs = both["dE_per_atom(b-a)"].abs()
        print(f"Mean |dE| per atom: {diffs.mean():.6f} eV   Max |dE| per atom: {diffs.max():.6f} eV")
        print(f"Mean signed dE per atom (b-a): {both['dE_per_atom(b-a)'].mean():.6f} eV")
    if (~ok).any():
        print("Not converged under both models (energies not directly comparable): "
              + ", ".join(table.loc[~ok, "material_id"]))

    table.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
