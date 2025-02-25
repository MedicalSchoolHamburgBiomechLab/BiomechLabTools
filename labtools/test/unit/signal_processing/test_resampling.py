import numpy as np

from labtools.signal_processing.resampling import resize_signal, convert_sample_rate


# Tests for resize_signal
def test_resize_signal_1d_increase_length():
    # Create a simple ramp signal
    sig = np.linspace(0, 1, 10)
    new_length = 20
    resized = resize_signal(sig, new_length)
    assert resized.shape[0] == new_length
    # Check that the first and last values remain approximately the same
    np.testing.assert_allclose(resized[0], sig[0], rtol=1e-5)
    np.testing.assert_allclose(resized[-1], sig[-1], rtol=1e-5)


def test_resize_signal_1d_decrease_length():
    sig = np.linspace(0, 1, 20)
    new_length = 10
    resized = resize_signal(sig, new_length)
    assert resized.shape[0] == new_length
    np.testing.assert_allclose(resized[0], sig[0], rtol=1e-5)
    np.testing.assert_allclose(resized[-1], sig[-1], rtol=1e-5)


def test_resize_signal_2d_axis0():
    # Create a 2D signal with shape (10, 3)
    sig = np.tile(np.linspace(0, 1, 10)[:, None], (1, 3))
    new_length = 15
    resized = resize_signal(sig, new_length, axis=0)
    assert resized.shape == (new_length, 3)
    np.testing.assert_allclose(resized[0, :], sig[0, :], rtol=1e-5)
    np.testing.assert_allclose(resized[-1, :], sig[-1, :], rtol=1e-5)


def test_resize_signal_2d_axis1():
    # Create a 2D signal with shape (10, 3)
    sig = np.tile(np.linspace(0, 1, 10)[:, None], (1, 3))
    new_length = 15
    resized = resize_signal(sig, new_length, axis=1)
    # The output is transposed relative to axis=0 resizing
    assert resized.shape == (10, new_length)
    np.testing.assert_allclose(resized[:, 0], sig[:, 0], rtol=1e-5)
    np.testing.assert_allclose(resized[:, -1], sig[:, -1], rtol=1e-5)


# Tests for convert_sample_rate
def test_convert_sample_rate_increase_rate():
    # Create a simple signal
    signal = np.linspace(0, 1, 10)
    f_in = 10
    f_out = 20
    resampled = convert_sample_rate(signal, f_in, f_out)
    # Expected new length: approximately len(signal) * (f_out / f_in)
    expected_length = int(np.ceil(len(signal) * (f_out / f_in)))
    assert resampled.shape[0] == expected_length
    np.testing.assert_allclose(resampled[0], signal[0], rtol=1e-5)
    np.testing.assert_allclose(resampled[-1], signal[-1], rtol=1e-5)


def test_convert_sample_rate_decrease_rate():
    signal = np.linspace(0, 1, 20)
    f_in = 20
    f_out = 10
    resampled = convert_sample_rate(signal, f_in, f_out)
    expected_length = int(np.ceil(len(signal) * (f_out / f_in)))
    assert resampled.shape[0] == expected_length
    np.testing.assert_allclose(resampled[0], signal[0], rtol=1e-5)
    np.testing.assert_allclose(resampled[-1], signal[-1], rtol=1e-5)
