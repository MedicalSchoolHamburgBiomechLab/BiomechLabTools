import numpy as np
from scipy import signal


def apply_butterworth(sig: np.array, sample_rate: int, cutoff: int, order: int = 4, btype: str = 'lowpass'):
    sos_lp = signal.butter(order, cutoff, btype=btype, fs=sample_rate, output='sos')
    sig_new = signal.sosfiltfilt(sos_lp, sig)
    return sig_new
