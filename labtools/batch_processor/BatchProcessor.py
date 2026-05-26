import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezc3d
import numpy as np
import pandas as pd
from scipy.io import loadmat


@dataclass
class BatchProcessor:
    path_root: Path
    file_types: str
    level_names: list[str]  # z.B. ["subject", "condition", "trial"]
    overwrite_output: bool = False
    path_output: Optional[Path] = field(default=None)
    _index: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _errors: Optional[list] = field(default=None, init=False, repr=False)
    skipped: list[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.path_output is None:
            self.path_output = self.path_root.parent / f"{self.path_root.stem}_processed"
        if self.path_output.exists() and not self.overwrite_output:
            raise FileExistsError(
                f"Output path {self.path_output} already exists. "
                f"Pass overwrite_output=True to use it anyway, or choose another path_output."
            )
        self._index = self._build_index()

    def _build_index(self) -> pd.DataFrame:
        rows = []
        self.skipped: list[Path] = []
        expected_depth = len(self.level_names)
        for fp in self.path_root.rglob(f"*{self.file_types}"):
            parts = fp.relative_to(self.path_root).parts
            labels = list(parts[:-1]) + [fp.stem]
            if len(labels) != expected_depth:
                self.skipped.append(fp)
                continue
            row = dict(zip(self.level_names, labels))
            row["path"] = fp
            rows.append(row)
        if self.skipped:
            preview = "\n  ".join(str(p) for p in self.skipped[:5])
            more = f"\n  ... (+{len(self.skipped) - 5} more)" if len(self.skipped) > 5 else ""
            warnings.warn(
                f"{len(self.skipped)} file(s) skipped (depth mismatch):\n  {preview}{more}\n"
                f"Full list available in self.skipped."
            )
        return pd.DataFrame(rows)

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def summary(self) -> dict:
        return {lvl: sorted(self._index[lvl].unique()) for lvl in self.level_names}

    def apply(self, func, multiprocess: bool = False, n_workers: Optional[int] = None, **kwargs) -> list:
        n = len(self._index)
        results: list = [None] * n
        errors = []

        if not multiprocess:
            for i, row in enumerate(self._index.itertuples()):
                try:
                    results[i] = func(row.path, **kwargs)
                except Exception as e:
                    errors.append((row.path, repr(e)))
        else:
            if n_workers is None:
                n_workers = 1 if sys.gettrace() else max(1, (os.cpu_count() or 2) - 1)

            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                future_to_idx = {
                    ex.submit(func, row.path, **kwargs): i
                    for i, row in enumerate(self._index.itertuples())
                }
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        errors.append((self._index.iloc[i].path, repr(e)))

        if errors:
            warnings.warn(f"{len(errors)} trial(s) failed. See bp.errors.")
            self._errors = errors
        return results

    @property
    def errors(self) -> list:
        return self._errors


def count_markers(path: Path) -> int:
    c = ezc3d.c3d(str(path))
    return c["parameters"]["POINT"]["USED"]["value"][0]


def unwrap(x):
    arr = np.asarray(x).ravel()
    return arr[0] if arr.size else np.nan


def read_mat_file(path: Path) -> dict:
    out = dict()
    data = loadmat(str(path))
    params = ['Stride_Length_Mean', 'Stride_Width_Mean', 'Steps_Per_Minute_Mean', 'Stance_Time_Mean_MEAN', 'Flight_Time_Mean_MEAN', 'Pelvis_Height_RANGE_cm_MEAN']
    for param in params:
        values = data.get(param)
        if values is not None:
            out[param] = np.nanmean(np.array([unwrap(cell) for cell in values.ravel()]))
        else:
            out[param] = np.nan
    return out


if __name__ == '__main__':
    path = Path(r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\AFT_Habituation_2\data\kinematics\mat")
    bp = BatchProcessor(path,
                        file_types="metrics.mat",
                        level_names=["participant", "session", "condition", "trial"])
    print(bp.summary())
    print(bp.skipped)

    results = bp.apply(read_mat_file,
                       # multiprocess=True
                       )

    # In case of dictionary result:
    metrics_df = pd.json_normalize(results)
    df = pd.concat([bp.index.reset_index(drop=True), metrics_df], axis=1)

    # In case of simple scalar result:
    # df = bp.index.assign(metrics=results)

    path_out_metrics_df = path.parent / "metrics.xlsx"
    df.to_excel(path_out_metrics_df)
    # ind = bp.index

    print(df.head())
    # print(df.groupby(["session", "condition", "trial"])["n_markers"].mean())
    print(bp.errors)
