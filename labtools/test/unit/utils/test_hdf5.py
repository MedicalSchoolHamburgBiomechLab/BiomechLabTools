import os
import tempfile

import numpy as np
import pytest

from labtools.utils.hdf5 import save_dict_to_hdf5, load_dict_from_hdf5


@pytest.fixture
def sample_dict():
    return {
        'array': np.array([1, 2, 3]),
        'array_of_strings': np.array(['a', 'b', 'c']),
        'int': 42,
        'float': 3.14,
        'string': 'Hello, World!',
        'bytes': b'byte string',
        'nested_dict': {
            'nested_array': np.array([[1, 2], [3, 4]]),
            'nested_int': np.int32(7),
        },
    }


def test_save_and_load_dict(sample_dict):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        filename = tmpfile.name

    try:
        # Save the dictionary to the HDF5 file
        save_dict_to_hdf5(sample_dict, filename)

        # Load the dictionary from the HDF5 file
        loaded_dict = load_dict_from_hdf5(filename)

        # Assert that the original and loaded dictionaries are equal
        assert_dict_equal(sample_dict, loaded_dict)

    finally:
        # Clean up the temporary file
        os.remove(filename)


def assert_dict_equal(d1, d2):
    assert d1.keys() == d2.keys(), "Keys mismatch"
    for key in d1:
        item1 = d1[key]
        item2 = d2[key]
        if isinstance(item1, dict):
            assert isinstance(item2, dict), f"Type mismatch for key '{key}'"
            assert_dict_equal(item1, item2)
        elif isinstance(item1, np.ndarray):
            assert isinstance(item2, np.ndarray), f"Type mismatch for key '{key}'"
            np.testing.assert_array_equal(item1, item2)
        else:
            assert item1 == item2, f"Value mismatch for key '{key}'"
