import numpy as np
import pandas as pd
import pytest

from labtools.utils.convenience import process_on_dataframe


@pytest.fixture
def df_sample():
    return pd.DataFrame({
        'A': [1, 2, 3, np.nan],
        'B': [4, 5, np.nan, 6],
        'C': [7, 8, 9, 10]
    })


def square(x):
    return x * x


def inverse(x):
    return 1 / x


def test_process_on_dataframe_single_thread(df_sample):
    # Expected result
    expected_result = pd.DataFrame({
        'A': [1, 4, 9, np.nan],
        'B': [16, 25, np.nan, 36],
        'C': [49, 64, 81, 100]
    })

    # Run the function without multiprocessing
    result = process_on_dataframe(df_sample, square, multiprocess=False)

    # Assert that the result matches the expected output
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_process_on_dataframe_multiprocessing(df_sample):
    # Expected result
    expected_result = pd.DataFrame({
        'A': [1, 4, 9, np.nan],
        'B': [16, 25, np.nan, 36],
        'C': [49, 64, 81, 100]
    })

    # Run the function with multiprocessing
    result = process_on_dataframe(df_sample, square, multiprocess=True)

    # Assert that the result matches the expected output
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_process_on_dataframe_exception_handling():
    # Create a DataFrame with invalid data
    df_sample = pd.DataFrame({
        'A': [1, 'two', 3],
        'B': [4, 5, 'six']
    })
    # Expected result with None where exceptions occurred
    expected_result = pd.DataFrame({
        'A': [1.0, None, 1 / 3],
        'B': [0.25, 0.2, None]
    })

    # Run the function without multiprocessing
    result = process_on_dataframe(df_sample, inverse, multiprocess=False)

    # Assert that the result matches the expected output
    pd.testing.assert_frame_equal(result, expected_result)
