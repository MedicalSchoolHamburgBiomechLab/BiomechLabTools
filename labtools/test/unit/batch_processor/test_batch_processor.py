"""Smoke tests for BatchProcessor.

Uses pytest's tmp_path fixture to build a synthetic data hierarchy and
verify indexing, summary, apply, error handling, and output-path
behaviour. Does not exercise multiprocess=True (would require a
top-level picklable function and adds little signal at this scope).
"""
import warnings
from pathlib import Path

import pytest

from labtools.batch_processor.batch_processor import BatchProcessor


@pytest.fixture
def hierarchy(tmp_path: Path) -> Path:
    """Build a small subject/condition/trial.dat tree under tmp_path."""
    layout = {
        ("S01", "A"): ["trial1.dat", "trial2.dat"],
        ("S01", "B"): ["trial1.dat"],
        ("S02", "A"): ["trial1.dat"],
        ("S03", "A"): ["trial1.dat"],
        ("S03", "B"): ["trial1.dat", "trial2.dat"],
    }
    root = tmp_path / "data"
    for (subj, cond), trials in layout.items():
        d = root / subj / cond
        d.mkdir(parents=True)
        for t in trials:
            (d / t).write_text("dummy")
    # One file at wrong depth — must be skipped.
    (root / "stray.dat").write_text("dummy")
    return root


def test_index_columns_and_size(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    assert list(bp.index.columns) == ["subject", "condition", "trial", "path"]
    assert len(bp.index) == 4


def test_depth_mismatch_is_skipped_with_warning(hierarchy):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bp = BatchProcessor(
            path_root=hierarchy,
            file_pattern=".dat",
            level_names=["subject", "condition", "trial"],
        )
    assert len(bp.skipped) == 1
    assert bp.skipped[0].name == "stray.dat"
    assert any("skipped" in str(warning.message) for warning in w)


def test_summary_returns_unique_values(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    s = bp.summary()
    assert s["subject"] == ["S01", "S02"]
    assert s["condition"] == ["A", "B"]
    assert "trial1" in s["trial"]


def test_apply_serial_returns_in_index_order(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    results = bp.apply(lambda p: p.name)
    assert results == [row.path.name for row in bp.index.itertuples()]
    assert bp.errors == []


def test_apply_catches_exceptions(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )

    def boom(path):
        raise ValueError(f"nope: {path.name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = bp.apply(boom)
    assert results == [None] * len(bp.index)
    assert len(bp.errors) == len(bp.index)
    assert all("ValueError" in repr(err) for _, err in bp.errors)


def test_errors_reset_between_apply_calls(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bp.apply(lambda p: 1 / 0)
        assert bp.errors  # populated from the failing run
        bp.apply(lambda p: 42)
    assert bp.errors == []


def test_default_output_path(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    assert bp.path_output == hierarchy.parent / f"{hierarchy.stem}_processed"
    assert not bp.path_output.exists()  # constructor must not create it


def test_ensure_output_dir_creates_path(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    out = bp.ensure_output_dir()
    assert out.exists() and out.is_dir()


def test_ensure_output_dir_raises_when_existing(hierarchy, tmp_path):
    existing = tmp_path / "already_here"
    existing.mkdir()
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
        path_output=existing,
    )
    with pytest.raises(FileExistsError):
        bp.ensure_output_dir()


def test_ensure_output_dir_allows_existing_when_flagged(hierarchy, tmp_path):
    existing = tmp_path / "already_here"
    existing.mkdir()
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
        path_output=existing,
        allow_existing_output=True,
    )
    assert bp.ensure_output_dir() == existing


def test_filter_remove_subject_inplace(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )

    to_remove = "S01"
    expected_remaining = set(bp.index.subject.unique()) - {to_remove}

    bp.filter(subject=[to_remove], method="remove", inplace=True)

    assert to_remove not in bp.index.subject.unique()
    assert set(bp.index.subject.unique()) == expected_remaining


def test_filter_keep_subject_inplace(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    s_ids_keep = ["S01"]
    bp.filter(subject=s_ids_keep, inplace=True)  # implicit "method='keep'"
    assert set(s_ids_keep) == set(bp.index.subject.unique())


def test_filter_returns_new_when_not_inplace(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    original_len = len(bp.index)
    bp_subset = bp.filter(subject=["S01"])

    # Returned a new object, original untouched.
    assert bp_subset is not bp
    assert len(bp.index) == original_len
    assert set(bp_subset.index.subject.unique()) == {"S01"}


def test_filter_inplace_returns_none(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    result = bp.filter(subject=["S01"], inplace=True)
    assert result is None


def test_filter_unknown_level_raises(hierarchy):
    bp = BatchProcessor(
        path_root=hierarchy,
        file_pattern=".dat",
        level_names=["subject", "condition", "trial"],
    )
    with pytest.raises(KeyError):
        bp.filter(nonexistent_level=["foo"])
