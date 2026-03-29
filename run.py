#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline test runner: point.py -> algorithem_og.py

This runs the 2-stage Hebrew entity resolution pipeline locally:
  Stage 1 (point.py):        Extracts person names from call transcripts
  Stage 2 (algorithem_og.py): Resolves names into canonical entities

Input:  CSV files in input/ folder
Output: CSV files in output/ folder + results printed to terminal

Usage:
    python run.py                          # Run full pipeline
    python run.py full                     # Same as above
    python run.py point                    # Run only name extraction
    python run.py algo                     # Run only entity resolution
    python run.py inspect                  # Show latest results
    python run.py inspect --name "אבו"     # Filter by name
    python run.py inspect --phone 5011     # Filter by phone
"""

import sys
import os
import io
import contextlib
import argparse
import pandas as pd

# ======================================================================
# CONFIGURATION
# All paths, column maps, and constants live here.
# ======================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEVEL_DIR = os.path.dirname(SCRIPT_DIR)

# Where the actual pipeline scripts live
POINT_DIR = os.path.join(TEVEL_DIR, "point to algorithm")
ALGO_DIR = os.path.join(TEVEL_DIR, "Algorithm")

# Where CSVs go
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def check_setup():
    """Verify pipeline scripts exist and dependencies are installed."""
    # Check pipeline scripts
    for name, dirpath, filename in [
        ("point.py", POINT_DIR, "point.py"),
        ("algorithem_og.py", ALGO_DIR, "algorithem_og.py"),
    ]:
        full = os.path.join(dirpath, filename)
        if not os.path.exists(full):
            print(f"\nERROR: Cannot find {name} at {full}")
            print(f"\n  This folder must be inside the tevel/ project:")
            print(f"    tevel/")
            print(f"    ├── test/                        ← you are here")
            print(f"    ├── point to algorithm/point.py")
            print(f"    └── Algorithm/algorithem_og.py")
            sys.exit(1)

    # Check Python dependencies
    missing = []
    for pkg in ["pandas", "numpy", "scipy", "networkx", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print(f"  Install: pip install -r requirements.txt")
        sys.exit(1)

# Point.py expects UUID column names in cube1.
# Fixture CSVs use human-readable names. This maps readable -> UUID.
POINT_UUID_COLUMNS = {
    "call_id":                 "6bfe4f0c-6176-48e0-a421-01dec3498f2e",
    "pstn_A":                  "3d79b534-559c-460c-88e8-05667cb83ed2",
    "pstn_B":                  "335d9912-987f-4453-8041-d059bfd6193b",
    "IntelliItemContent":      "IntelliItemContent",
    "IntelliItemCreationTime": "IntelliItemCreationTime",
}

# Point output -> Algorithm input column rename
# (Algorithm's _adapt_schema also does this, but we save the intermediate)
POINT_TO_ALGO_RENAME = {
    "original_call_id":    "call_id",
    "num_A":               "pstn_A",
    "num_B":               "pstn_B",
    "name_cleaned_A":      "clean_name_A",
    "name_cleaned_B":      "clean_name_B",
    "verified_nickname_A": "verified_nicknames_A",
    "verified_nickname_B": "verified_nicknames_B",
}

# Columns to drop (algorithm doesn't need them)
POINT_DROP_COLUMNS = [
    "fake_id", "name+role_A", "name+role_B", "role_A", "role_B",
    "extraction_method", "extraction_status", "metadata",
]


# ======================================================================
# LOADING FUNCTIONS
# ======================================================================

def load_csv(filename, required=True):
    """Load a CSV from input/. Returns DataFrame or None if optional and missing."""
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        if required:
            available = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
            print(f"\nERROR: Cannot find {path}")
            print(f"  This file is REQUIRED.")
            print(f"  Available files in input/: {available or '(none)'}")
            sys.exit(1)
        return None
    df = pd.read_csv(path)
    return df


def load_point_cubes():
    """Load all input CSVs and build the cubes dict for point.py.
    Returns dict like {'cube1': df, 'cube5': df5, ...}
    """
    # Load cube1 (call data) - REQUIRED
    cube1 = load_csv("point_cube1.csv", required=True)

    # Auto-detect: if UUID columns already present, skip remapping
    uuid_col = "6bfe4f0c-6176-48e0-a421-01dec3498f2e"
    if uuid_col not in cube1.columns:
        # Human-readable column names -> remap to UUIDs
        rename = {k: v for k, v in POINT_UUID_COLUMNS.items() if k in cube1.columns}
        cube1 = cube1.rename(columns=rename)

    # Load cube5 (config) - REQUIRED
    cube5 = load_csv("point_cube5.csv", required=True)

    # Load optional cubes
    cube3 = load_csv("point_cube3.csv", required=False)
    cube4 = load_csv("point_cube4.csv", required=False)

    cubes = {"cube1": cube1, "cube5": cube5}
    if cube3 is not None:
        cubes["cube3"] = cube3
    if cube4 is not None:
        cubes["cube4"] = cube4

    return cubes


def load_algo_cube2():
    """Load optional phonebook CSV for algorithm. Returns DataFrame or None."""
    return load_csv("algo_cube2.csv", required=False)


# ======================================================================
# COLUMN MAPPING
# ======================================================================

def map_point_to_algo(point_df):
    """Rename point output columns to algorithm input columns. Drop extras.
    Returns a new DataFrame ready for algorithm.
    """
    df = point_df.copy()

    # Rename columns (only if target doesn't already exist)
    rename = {old: new for old, new in POINT_TO_ALGO_RENAME.items()
              if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename)

    # Drop diagnostic columns the algorithm doesn't need
    drop = [c for c in POINT_DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)

    return df


# ======================================================================
# RUNNING FUNCTIONS
# ======================================================================

def import_point():
    """Import point.py from its directory. Returns the module."""
    # Clear cached module so re-runs pick up edits
    if "point" in sys.modules:
        del sys.modules["point"]
    sys.path.insert(0, POINT_DIR)
    try:
        import point as mod
    finally:
        sys.path.pop(0)
    return mod


def import_algorithm():
    """Import algorithem_og.py from its directory. Returns the module."""
    if "algorithem_og" in sys.modules:
        del sys.modules["algorithem_og"]
    sys.path.insert(0, ALGO_DIR)
    try:
        import algorithem_og as mod
    finally:
        sys.path.pop(0)
    return mod


def run_point_stage():
    """Run point.py on input CSVs. Returns output DataFrame."""
    cubes = load_point_cubes()
    print_header("STAGE 1: point.py (name extraction)")

    cube1 = cubes["cube1"]
    cube5 = cubes["cube5"]
    main_num = cube5.iloc[0].get("SHLUHA", "?") if not cube5.empty else "?"
    print(f"  Input:  {len(cube1)} calls from input/point_cube1.csv")
    print(f"  Main #: {main_num}")
    if "cube3" in cubes:
        print(f"  Cube3:  {len(cubes['cube3'])} entity records")
    if "cube4" in cubes:
        print(f"  Cube4:  {len(cubes['cube4'])} script1 records")

    point_mod = import_point()

    # Suppress point.py's debug output — save to log file instead
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        result = point_mod.main(cubes)

    # Save log
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "point_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buf.getvalue())

    # Check for error
    if "error" in result.columns:
        print(f"\n  ERROR from point.py: {result['error'].iloc[0]}")
        print(f"  See log: {log_path}")
        sys.exit(1)

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "point_output.csv")
    result.to_csv(out_path, index=False)

    # Show extraction stats
    if "extraction_status" in result.columns:
        success = (result["extraction_status"] == "success").sum()
        failed = (result["extraction_status"] == "failed").sum()
        print(f"  Result: {len(result)} rows ({success} success, {failed} failed)")
    else:
        print(f"  Result: {len(result)} rows ({len(result.columns)} columns)")
    print(f"  Saved:  {out_path}")
    print(f"  Log:    {log_path}")

    return result


def run_mapping_stage(point_df):
    """Map point output columns to algorithm input. Returns mapped DataFrame."""
    print_header("COLUMN MAPPING")

    mapped = map_point_to_algo(point_df)

    # Show what was renamed
    renamed = {old: new for old, new in POINT_TO_ALGO_RENAME.items()
               if old in point_df.columns and new not in point_df.columns}
    if renamed:
        pairs = ", ".join(f"{o}->{n}" for o, n in renamed.items())
        print(f"  Renamed: {pairs}")

    dropped = [c for c in POINT_DROP_COLUMNS if c in point_df.columns]
    if dropped:
        print(f"  Dropped: {', '.join(dropped)}")

    # Save intermediate
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "algo_input.csv")
    mapped.to_csv(out_path, index=False)
    print(f"  Saved:  {out_path} ({len(mapped.columns)} columns)")

    return mapped


def run_algo_stage(algo_input_df=None):
    """Run algorithem_og.py. Returns output DataFrame."""
    print_header("STAGE 2: algorithem_og.py (entity resolution)")

    # If no input provided, try to load saved algo_input.csv
    if algo_input_df is None:
        saved_path = os.path.join(OUTPUT_DIR, "algo_input.csv")
        if not os.path.exists(saved_path):
            # Try point_output.csv as fallback (algorithm has _adapt_schema)
            saved_path = os.path.join(OUTPUT_DIR, "point_output.csv")
        if not os.path.exists(saved_path):
            print(f"\n  ERROR: No algorithm input found.")
            print(f"  Run the full pipeline first: python run.py")
            sys.exit(1)
        algo_input_df = pd.read_csv(saved_path)
        print(f"  Loaded: {saved_path}")

    # Build cubes dict
    cubes = {"cube1": algo_input_df}
    cube2 = load_algo_cube2()
    if cube2 is not None:
        cubes["cube2"] = cube2
        print(f"  Cube2:  {len(cube2)} phonebook contacts")

    phones = set()
    for col in ["pstn_A", "pstn_B", "speaker_phone"]:
        if col in algo_input_df.columns:
            phones.update(algo_input_df[col].dropna().unique())
    print(f"  Input:  {len(algo_input_df)} rows, {len(phones)} phones")

    algo_mod = import_algorithm()

    # Suppress algorithm's debug output — save to log file instead
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        result = algo_mod.main(cubes)

    # Save log
    log_path = os.path.join(OUTPUT_DIR, "algo_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buf.getvalue())

    # Check for error
    if "error" in result.columns:
        print(f"\n  ERROR from algorithm: {result['error'].iloc[0]}")
        print(f"  See log: {log_path}")
        sys.exit(1)

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "algo_output.csv")
    result.to_csv(out_path, index=False)

    # Summary stats
    if "resolved_type" in result.columns:
        type_counts = result["resolved_type"].value_counts()
        types_str = ", ".join(f"{cnt} {typ}" for typ, cnt in type_counts.items())
        print(f"  Result: {types_str}")
    else:
        print(f"  Result: {len(result)} rows")

    # Show cross-phone links
    if "global_entity_id" in result.columns and "speaker_phone" in result.columns:
        multi = result.groupby("global_entity_id")["speaker_phone"].nunique()
        cross_count = (multi > 1).sum()
        print(f"  Cross-phone entities: {cross_count}")

    print(f"  Saved:  {out_path} ({len(result.columns)} columns)")
    print(f"  Log:    {log_path}")

    return result


# ======================================================================
# PRINTING FUNCTIONS
# ======================================================================

def print_header(text):
    """Print a stage header."""
    line = "=" * 50
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def print_results_table(df, name_filter=None, phone_filter=None, max_rows=40):
    """Print a readable results table from algorithm output."""
    if df.empty:
        print("  (no results)")
        return

    # Determine which columns to show
    show_cols = []
    for col in ["speaker_phone", "global_entity_id", "resolved_name",
                 "resolved_type", "verified_entity_id", "confidence"]:
        if col in df.columns:
            show_cols.append(col)

    if not show_cols:
        print("  (no recognized output columns)")
        return

    view = df[show_cols].copy()

    # Apply filters
    if name_filter:
        mask = pd.Series(False, index=view.index)
        for col in ["resolved_name", "raw_clean_name", "verified_name_call"]:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(name_filter, na=False)
        view = view[mask]
        print(f"  (filtered by name: '{name_filter}' -> {len(view)} rows)")

    if phone_filter:
        mask = pd.Series(False, index=view.index)
        for col in ["speaker_phone", "other_phone"]:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(phone_filter, na=False)
        view = view[mask]
        print(f"  (filtered by phone: '{phone_filter}' -> {len(view)} rows)")

    if view.empty:
        print("  (no matching rows)")
        return

    # Truncate
    if len(view) > max_rows:
        print(f"  (showing first {max_rows} of {len(view)} rows)")
        view = view.head(max_rows)

    # Print table
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 120)
    pd.set_option("display.unicode.east_asian_width", True)
    print()
    print(view.to_string(index=False))
    print()


# ======================================================================
# MAIN
# ======================================================================

def cmd_full(args):
    """Run the full pipeline: point -> map -> algorithm."""
    point_df = run_point_stage()
    mapped_df = run_mapping_stage(point_df)
    algo_df = run_algo_stage(mapped_df)
    print_header("RESULTS")
    print_results_table(algo_df, name_filter=args.name, phone_filter=args.phone)


def cmd_point(args):
    """Run only point.py."""
    point_df = run_point_stage()
    print_header("POINT OUTPUT (first 10 rows)")
    cols = ["num_A", "num_B", "name_cleaned_A", "name_cleaned_B",
            "extraction_status", "verified_entity_id_A"]
    show = [c for c in cols if c in point_df.columns]
    if show:
        print()
        print(point_df[show].head(10).to_string(index=False))
        print()


def cmd_algo(args):
    """Run only algorithm (uses saved point output)."""
    algo_df = run_algo_stage()
    print_header("RESULTS")
    print_results_table(algo_df, name_filter=args.name, phone_filter=args.phone)


def cmd_inspect(args):
    """Show latest algorithm output."""
    out_path = os.path.join(OUTPUT_DIR, "algo_output.csv")
    if not os.path.exists(out_path):
        print(f"\nNo output found at {out_path}")
        print(f"Run the pipeline first: python run.py")
        sys.exit(1)

    df = pd.read_csv(out_path)
    print_header("RESULTS (from saved output)")
    print(f"  File: {out_path}")
    print(f"  Rows: {len(df)}")
    if "resolved_type" in df.columns:
        type_counts = df["resolved_type"].value_counts()
        types_str = ", ".join(f"{cnt} {typ}" for typ, cnt in type_counts.items())
        print(f"  Types: {types_str}")
    print_results_table(df, name_filter=args.name, phone_filter=args.phone)


def main():
    check_setup()
    parser = argparse.ArgumentParser(
        description="Pipeline test: point.py -> algorithem_og.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Run full pipeline
  python run.py point              Run only point.py
  python run.py algo               Run only algorithm
  python run.py inspect            Show latest results
  python run.py inspect --name "אבו"   Filter by name
""")
    parser.add_argument("action", nargs="?", default="full",
                        choices=["full", "point", "algo", "inspect"],
                        help="What to run (default: full)")
    parser.add_argument("--name", default=None,
                        help="Filter results by name substring")
    parser.add_argument("--phone", default=None,
                        help="Filter results by phone substring")
    args = parser.parse_args()

    if args.action == "full":
        cmd_full(args)
    elif args.action == "point":
        cmd_point(args)
    elif args.action == "algo":
        cmd_algo(args)
    elif args.action == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
