import numpy as np
from scipy.interpolate import interp1d


def resize_signal(signal: np.ndarray, new_length: int, axis: int = 0) -> np.ndarray:
    """
    Resize a signal to a new length using interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array. Can be 1D or 2D.
    new_length : int
        Desired length along the specified axis.
    axis : int, optional
        Axis along which to resize the signal (default is 0). If set to 1, the signal is transposed,
        resized along axis 0, and then transposed back.

    Returns
    -------
    np.ndarray
        The resized signal.

    Notes
    -----
    The function uses linear interpolation to compute the new sample values.
    """
    sig = signal.T.copy() if axis == 1 else signal.copy()
    if len(sig) == new_length:
        return sig.T if axis == 1 else sig
    x = np.arange(len(sig))
    new_x = np.linspace(0, x[-1], new_length)
    if sig.ndim == 1:
        out = interp1d(x, sig)(new_x)
    else:
        out = np.column_stack([interp1d(x, sig[:, i])(new_x) for i in range(sig.shape[1])])
    return out.T if axis == 1 else out


def convert_sample_rate(signal: np.ndarray, f_in: int, f_out: int) -> np.ndarray:
    """
    Convert the sample rate of a 1D signal using linear interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal array.
    f_in : int
        Original sampling rate (samples per second).
    f_out : int
        Desired sampling rate (samples per second).

    Returns
    -------
    np.ndarray
        The signal resampled to the new sampling rate.

    Notes
    -----
    The function calculates new sample positions based on the ratio f_in / f_out
    and uses np.interp for linear interpolation to compute the resampled signal.
    The new sample positions include the last sample of the original signal.
    """
    original_length = len(signal)
    expected_length = int(np.ceil(original_length * (f_out / f_in)))
    x_old = np.arange(original_length)
    x_new = np.linspace(0, original_length - 1, expected_length)
    return np.interp(x_new, x_old, signal)
