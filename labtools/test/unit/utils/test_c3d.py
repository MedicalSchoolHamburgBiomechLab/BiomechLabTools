from pathlib import Path

import pytest

from labtools.utils.c3d import load_c3d


@pytest.fixture
def c3d_data():
    # Get the directory where this test file is located
    test_dir = Path(__file__).parent

    # Construct the path to the data file relative to the test file
    file_path = test_dir / ".." / ".." / "data" / "c3d_testfile.c3d"
    file_path = file_path.resolve()

    assert file_path.exists(), f"File not found: {file_path}"

    data, meta = load_c3d(file_path)
    return data


def test_analog_rate(c3d_data):
    assert c3d_data['analog_rate'] == 0.0


def test_point_rate(c3d_data):
    assert c3d_data['point_rate'] == 300


def test_marker_shapes(c3d_data):
    assert c3d_data['marker']['SIPS_left'].shape == (240, 3)
    assert c3d_data['marker']['SIPS_right'].shape == (240, 3)


def test_analog_shape(c3d_data):
    assert c3d_data['analog'].shape == (240, 0)


def test_creation_date(c3d_data):
    assert c3d_data['creation_date'] == '2023-05-04'


def test_creation_time(c3d_data):
    assert c3d_data['creation_time'] == '10:53:24'
