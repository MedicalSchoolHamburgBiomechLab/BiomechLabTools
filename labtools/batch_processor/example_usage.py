"""
Example usage of BatchProcessor.

Builds a synthetic data hierarchy in a temporary directory, then
demonstrates the two most common result patterns:
- scalar return  -> attach via DataFrame.assign
- dict return    -> expand to columns via pd.json_normalize

Run directly:

    python example_usage.py

For real-world use against your own data, point ``path_root`` to your
data directory and replace the demo processor functions with whatever
you actually need (read C3D, MAT, EDF, hdf5, parquet, ...).
"""
import tempfile
from pathlib import Path

import pandas as pd

from batch_processor import BatchProcessor


# --- demo processor functions ---------------------------------------------
#
# Replace these with real ones (e.g. ezc3d.c3d(path), scipy.io.loadmat,
# pyedflib, ...). The contract is simple: take a Path, return whatever.

def file_size_bytes(path: Path) -> int:
    """Return the size of a file in bytes.

    Demonstrates the scalar-return pattern: one number per trial,
    attached to the index via ``DataFrame.assign``.
    """
    return path.stat().st_size


def file_stats(path: Path) -> dict:
    """Return several statistics about a file as a dict.

    Demonstrates the dict-return pattern: multiple values per trial,
    expanded into separate columns via ``pd.json_normalize``.
    """
    stat = path.stat()
    text = path.read_text()
    return {
        "size_bytes": stat.st_size,
        "n_lines": text.count("\n") + 1,
        "n_chars": len(text),
    }


# --- synthetic data setup -------------------------------------------------

def build_demo_hierarchy(root: Path) -> None:
    """Create a small subject/session/condition/trial.txt tree under root."""
    layout = {
        ("S01", "PRE", "A"): ["trial1", "trial2"],
        ("S01", "POST", "A"): ["trial1"],
        ("S02", "PRE", "A"): ["trial1", "trial2", "trial3"],
        ("S02", "PRE", "B"): ["trial1"],
    }
    for (subj, sess, cond), trials in layout.items():
        d = root / subj / sess / cond
        d.mkdir(parents=True, exist_ok=True)
        for t in trials:
            (d / f"{t}.txt").write_text(
                f"demo content for {subj}/{sess}/{cond}/{t}\n" * 3
            )


# --- main -----------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        path_root = Path(tmp) / "demo_data"
        build_demo_hierarchy(path_root)

        bp = BatchProcessor(
            path_root=path_root,
            file_pattern=".txt",
            level_names=["participant", "session", "condition", "trial"],
        )

        print("Index summary:")
        for level, values in bp.summary().items():
            print(f"  {level}: {values}")
        print(f"Skipped: {bp.skipped}\n")

        # Scalar-return pattern.
        sizes = bp.apply(file_size_bytes)
        df_scalar = bp.index.assign(size_bytes=sizes)
        print("Scalar-return result (head):")
        print(df_scalar.head(), "\n")

        # Dict-return pattern.
        stats = bp.apply(file_stats)
        stats_df = pd.json_normalize(stats)
        df_dict = pd.concat([bp.index.reset_index(drop=True), stats_df], axis=1)
        print("Dict-return result (head):")
        print(df_dict.head(), "\n")

        print(f"Errors: {bp.errors}")
