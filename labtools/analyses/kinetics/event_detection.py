import numpy as np
from scipy.signal import find_peaks

from labtools.signal_analysis.filtering import apply_butterworth


def get_foot_events(f_z, sample_rate, threshold: float = 10.0):
    arg_max = np.argmax(f_z)
    if len(np.where(f_z[:arg_max] < threshold)[0]) < 2:
        raise ValueError('Not enough data points below threshold before maximum')
    ic_frame = get_ic_frame(fz=f_z, sample_rate=sample_rate, threshold=threshold)
    tc_frame = get_tc_frame(f_z, ic_frame, threshold=threshold)
    return ic_frame, tc_frame


def get_ic_frame(fz: np.ndarray, sample_rate: int, threshold: float = 10.0):
    window = int(sample_rate / 10)  # 10th of a second as window
    f_z_len = len(fz)
    for ic in range(f_z_len - window + 1):
        if fz[ic] >= threshold and np.all(fz[ic:(ic + window)] > threshold):
            return ic - 1

    return -1


def get_tc_frame(fz: np.ndarray, ic_frame: int, threshold: float = 10.0):
    for tc in range(len(fz) - 1, ic_frame, -1):
        if fz[tc] < threshold:
            subarray = fz[(ic_frame + 1):tc]
            if np.all(subarray >= threshold) and len(subarray) > 0:
                return tc + 1
    return -1


def get_midpoints(f_z, peaks, threshold):
    mid = []
    for p1, p2 in zip(peaks[:-1], peaks[1:]):
        window = f_z[p1:p2]
        below_threshold = np.where(window < threshold)[0]
        if len(below_threshold) < 3:
            if (p1 != 0) & (p2 != (len(f_z) - 1)):
                print('too few data points')
            continue
        mid.append(np.median(below_threshold).astype(int) + p1)
    return mid


def get_force_events_treadmill(f_z: np.ndarray, sample_rate: int, threshold: float = 10.0, offset_corr: bool = False):
    step_freq = 3.5
    dist = (0.8 * sample_rate) / step_freq
    min_height = 100
    fz_filt = apply_butterworth(sig=f_z, sample_rate=sample_rate, cutoff=20)
    peaks, _ = find_peaks(fz_filt, distance=dist, height=min_height)
    peaks = np.hstack(([0], peaks, [len(f_z) - 1]))

    mid = get_midpoints(f_z, peaks, threshold)

    ic_list = list()
    tc_list = list()
    for m1, m2 in zip(mid[:-1], mid[1:]):
        step = f_z[m1:m2]
        if offset_corr:
            step -= np.mean(np.sort(step)[:10])
        try:
            ic, tc = get_foot_events(step, sample_rate=sample_rate, threshold=threshold)
            ic_list.append(ic + m1)
            tc_list.append(tc + m1)
        except Exception as e:
            print(f'get_foot_events failed: ', e)
    return {'ic': ic_list, 'tc': tc_list}
