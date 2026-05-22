from dataclasses import dataclass

import numpy as np
from ezc3d import c3d


class TheiaFormatError(ValueError):
    """Raised when a Theia C3D file's structure doesn't match expectations."""


EXPECTED_INERTIA_DESCRIPTOR = [
    "length",
    "mass percentage female",
    "COM female (x, y, z)",
    "radius of gyration female (x, y, z)",
    "mass percentage male",
    "COM male (x, y, z)",
    "radius of gyration male (x, y, z)",
]

SUPPORTED_THEIA3D_VERSIONS: set[tuple[int, ...]] = {
    (2025, 1, 2),
}

SUPPORTED_MODEL_VERSIONS: set[tuple[int, ...]] = {
    (12, 0, 0),
}


@dataclass
class Segment:
    name: str
    proximal_pos: np.ndarray  # (3, n_frames) global
    distal_pos: np.ndarray  # (3, n_frames) global
    rotation: np.ndarray  # (3, 3, n_frames) LCS -> global
    length: float  # m, aus THEIA3D-Parametern
    mass_fraction: float | None
    com_local: np.ndarray | None  # (3,) im LCS
    landmarks_local: dict[str, np.ndarray]  # {"heel": [...], "toe_tip": [...]}

    # Convenience
    @property
    def com_global(self) -> np.ndarray:
        return self.proximal_pos + np.einsum("ijt,j->it", self.rotation, self.com_local)


@dataclass
class Trial:
    sample_rate: float
    segments: dict[str, Segment]  # 'l_thigh', 'pelvis', ...
    metadata: dict  # source, theia_version, model_version, n_frames


def _check_inertia_descriptor(descriptor: list[str]) -> None:
    if len(descriptor) != 1:
        raise TheiaFormatError(
            f"Expected SEGMENT_INERTIA_PARAMETERS to be a list of length 1, "
            f"got length {len(descriptor)}."
        )
    parts = descriptor[0].split("; ")
    if parts != EXPECTED_INERTIA_DESCRIPTOR:
        raise TheiaFormatError(
            "SEGMENT_INERTIA_PARAMETERS descriptor does not match expected format.\n"
            f"  Expected: {EXPECTED_INERTIA_DESCRIPTOR}\n"
            f"  Got:      {parts}"
        )


def _get_version(theia_params: dict, key: str) -> tuple[int, ...]:
    """Extract a version tuple from the THEIA3D parameter group."""
    if key not in theia_params:
        raise TheiaFormatError(f"Missing {key} in THEIA3D parameters.")
    val = np.asarray(theia_params[key]["value"]).ravel()
    return tuple(int(x) for x in val)


def _check_supported_version(
        version: tuple[int, ...],
        supported: set[tuple[int, ...]],
        name: str,
) -> None:
    if version not in supported:
        raise TheiaFormatError(
            f"Unsupported {name}: {version}. "
            f"Supported versions: {sorted(supported)}. "
            f"Either add this version to the supported set after verifying "
            f"compatibility, or update the reader."
        )


def read_theia_c3d(path) -> Trial:
    acq = c3d(path)
    # get the theia meta data
    theia_params = acq["parameters"].get("THEIA3D")  # dict with all theia parameters
    if theia_params is None:
        raise ValueError("This C3D file does not contain THEIA3D parameters. Please check the file and try again.")
    metadata = {}
    theia_version = _get_version(theia_params, "THEIA3D_VERSION")
    model_version = _get_version(theia_params, "MODEL_VERSION")
    _check_supported_version(theia_version, SUPPORTED_THEIA3D_VERSIONS, "Theia3D version")
    _check_supported_version(model_version, SUPPORTED_MODEL_VERSIONS, "Theia model version")

    metadata.update({
        "theia_version": theia_version,
        "model_version": model_version,
        "source": "theia_c3d",
        "source_path": str(path),
    })

    seg_inertial_params = theia_params.get("SEGMENT_INERTIA_PARAMETERS")
    if seg_inertial_params is None:
        raise ValueError("This C3D file does not contain THEIA3D SEGMENT_INERTIA_PARAMETERS. Please check the file and try again.")
    value_descriptor = seg_inertial_params["value"]
    # this will raise a TheiaFormatError if the descriptor does not match expectations
    _check_inertia_descriptor(value_descriptor)
    # get the segments and their data
    rotation_data = acq["data"]["rotations"]
    labels = acq["parameters"]["ROTATION"]["LABELS"]["value"]
    for l, label in enumerate(labels):
        if not "pelv" in label.lower():
            continue
        # get the rotation data for this segment. The shape of rotation_data is (4, 4, n_segments, n_frames), so we index with l to get the data for this segment:
        rot_pos = rotation_data[:, :, l, :]
        rot = rot_pos[:3, :3, :]
        prox_pos = rot_pos[:3, 3, :]


if __name__ == '__main__':
    path = r"C:\Users\dominik.fohrmann\Downloads\pose_filt_0.c3d"

    trial = read_theia_c3d(path)
