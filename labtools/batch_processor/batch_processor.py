"""
BatchProcessor: a lightweight container for iterating over hierarchically
organised data files.

Given a root directory with a consistent folder structure (e.g.
subject/condition/trial.ext), BatchProcessor builds an index and applies
a user-supplied function to each trial, optionally in parallel.

The class is intentionally agnostic about file format or processing:
user functions receive a file path and return whatever they want.
Collection and post-processing of results is the caller's responsibility.
"""
import copy
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


@dataclass
class BatchProcessor:
    """Index and process files in a fixed-depth folder hierarchy.

    Walks ``path_root`` for files matching ``file_pattern``, builds a
    DataFrame index whose columns are taken from ``level_names``, and
    exposes an :meth:`apply` method that runs a user-defined function on
    each indexed file.

    Files whose folder depth does not match ``len(level_names)`` are
    skipped and recorded in :attr:`skipped`. Exceptions raised inside the
    user function during :meth:`apply` are caught and recorded in
    :attr:`errors` rather than aborting the run.

    The class is purely an indexer and dispatcher; it does not load or
    write data. ``path_output`` is provided as a convention for user
    functions to consult and is never written to by the class itself.
    Call :meth:`ensure_output_dir` before writing to it.

    Parameters
    ----------
    path_root : Path
        Root directory of the data hierarchy.
    file_pattern : str
        Suffix or pattern matched by ``rglob(f"*{file_pattern}")``
        (e.g. ``".c3d"``, ``"metrics.mat"``).
    level_names : list of str
        Names of the hierarchy levels, ordered from outermost (closest
        to root) to innermost. The innermost level corresponds to the
        file stem, not a folder. Example:
        ``["subject", "session", "condition", "trial"]``.
    path_output : Path, optional
        Suggested output location for user functions. Defaults to
        ``<path_root>_processed`` next to ``path_root``. The class
        itself never writes to this path.
    allow_existing_output : bool, default False
        If False, :meth:`ensure_output_dir` raises
        :class:`FileExistsError` when ``path_output`` already exists.
        Construction does not check the path; the check happens only
        when output is actually needed.

    Attributes
    ----------
    index : pd.DataFrame
        Read-only view of the indexed files. Columns are
        ``level_names`` plus a ``path`` column.
    skipped : list of Path
        Files found by rglob but skipped due to depth mismatch.
    errors : list of (Path, str)
        Files that raised an exception during the most recent
        :meth:`apply` call. Reset at the start of each call.
    """
    path_root: Path
    file_pattern: str
    level_names: list[str]
    path_output: Optional[Path] = None
    allow_existing_output: bool = False
    _index: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _errors: list = field(default_factory=list, init=False, repr=False)
    _skipped: list[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.path_output is None:
            self.path_output = (
                    self.path_root.parent / f"{self.path_root.stem}_processed"
            )
        self._index = self._build_index()

    def _build_index(self) -> pd.DataFrame:
        """Walk ``path_root`` and build the file index.

        Files whose folder depth does not match ``len(level_names)`` are
        appended to :attr:`skipped` and excluded from the index. A
        warning is emitted if any files were skipped.
        """
        rows: list[dict] = []
        self._skipped = []
        expected_depth = len(self.level_names)

        for fp in self.path_root.rglob(f"*{self.file_pattern}"):
            parts = fp.relative_to(self.path_root).parts
            labels = list(parts[:-1]) + [fp.stem]
            if len(labels) != expected_depth:
                self._skipped.append(fp)
                continue
            row = dict(zip(self.level_names, labels))
            row["path"] = fp
            rows.append(row)

        if self._skipped:
            preview = "\n  ".join(str(p) for p in self._skipped[:5])
            more = (
                f"\n  ... (+{len(self._skipped) - 5} more)"
                if len(self._skipped) > 5 else ""
            )
            warnings.warn(
                f"{len(self._skipped)} file(s) skipped (depth mismatch):\n"
                f"  {preview}{more}\n"
                f"Full list available in self.skipped."
            )
        return pd.DataFrame(rows)

    @property
    def index(self) -> pd.DataFrame:
        """Indexed files as a DataFrame (read-only view)."""
        return self._index

    @property
    def errors(self) -> list:
        """``(path, repr(exception))`` tuples from the most recent :meth:`apply`."""
        return self._errors

    @property
    def skipped(self) -> list[Path]:
        """Files skipped during indexing due to depth mismatch."""
        return self._skipped

    def summary(self) -> dict[str, list]:
        """Return a dict of unique values per level.

        Useful as a quick sanity check after construction: does the
        index contain the participants, sessions, conditions you
        expected?
        """
        return {lvl: sorted(self._index[lvl].unique()) for lvl in self.level_names}

    def ensure_output_dir(self) -> Path:
        """Create :attr:`path_output` if needed and return it.

        Call this from user functions that write output. Raises
        :class:`FileExistsError` if the directory already exists and
        ``allow_existing_output`` is False.
        """
        if self.path_output.exists() and not self.allow_existing_output:
            raise FileExistsError(
                f"Output path {self.path_output} already exists. "
                f"Pass allow_existing_output=True to use it anyway, "
                f"or choose another path_output."
            )
        self.path_output.mkdir(parents=True, exist_ok=True)
        return self.path_output

    def apply(
            self,
            func: Callable[..., Any],
            multiprocess: bool = False,
            n_workers: Optional[int] = None,
            **kwargs,
    ) -> list:
        """Apply ``func`` to every file in the index.

        The function is called as ``func(path, **kwargs)`` and its
        return value is collected into a list whose order matches
        :attr:`index`. Exceptions raised inside ``func`` are caught and
        recorded in :attr:`errors`; the corresponding result entry is
        ``None``.

        Parameters
        ----------
        func : callable
            Function taking a :class:`pathlib.Path` as its first
            positional argument and returning anything. Must be a
            top-level (picklable) function when ``multiprocess=True``.
        multiprocess : bool, default False
            Run in parallel via :class:`ProcessPoolExecutor`. Default is
            serial, which is easier to debug and often faster for
            lightweight functions (the IPC overhead dominates).
        n_workers : int, optional
            Number of worker processes. Defaults to ``cpu_count() - 1``,
            or 1 if a debugger is attached.
        **kwargs
            Forwarded to ``func``.

        Returns
        -------
        list
            Results in the same order as :attr:`index`. Failed trials
            yield ``None`` at their position.

        Notes
        -----
        When ``multiprocess=True`` on Windows, the calling script must
        be guarded by ``if __name__ == "__main__":`` to avoid recursive
        process spawning.
        """
        n = len(self._index)
        results: list = [None] * n
        self._errors = []  # reset per call

        if not multiprocess:
            for i, row in enumerate(self._index.itertuples()):
                try:
                    results[i] = func(row.path, **kwargs)
                except Exception as e:
                    self._errors.append((row.path, repr(e)))
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
                        self._errors.append((self._index.iloc[i].path, repr(e)))

        if self._errors:
            warnings.warn(f"{len(self._errors)} trial(s) failed. See self.errors.")
        return results

    def filter(
            self,
            method: str = "keep",
            inplace: bool = False,
            **criteria,
    ) -> Optional["BatchProcessor"]:
        """Filter the index by level values.

        Parameters
        ----------
        method : {"keep", "remove"}, default "keep"
            Whether the criteria specify which rows to keep or remove.
        inplace : bool, default False
            If True, modify this BatchProcessor in place and return None.
            If False, return a new BatchProcessor and leave this one unchanged.
        **criteria
            Level name -> value or list of values. Combined with AND.
        """
        if method not in ("keep", "remove"):
            raise ValueError(f"method must be 'keep' or 'remove', got {method!r}")

        mask = pd.Series(True, index=self._index.index)
        for level, values in criteria.items():
            if level not in self._index.columns:
                raise KeyError(f"Unknown level: {level!r}. Available: {self.level_names}")
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            mask &= self._index[level].isin(values)
        if method == "remove":
            mask = ~mask

        filtered_index = self._index[mask].reset_index(drop=True)

        if inplace:
            self._index = filtered_index
            self._errors = []
            return None

        new = copy.copy(self)
        new._index = filtered_index
        new._errors = []
        return new
