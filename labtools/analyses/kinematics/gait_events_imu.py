import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

from labtools.utils.convenience import sample_rate_from_timestamp


def get_cadence_estimate(gyro_pitch: np.ndarray, sample_rate: int):
    assert gyro_pitch.ndim == 1
    n_seconds_max = 60

    # Trim the signal if it's too long
    if len(gyro_pitch) > n_seconds_max * sample_rate:
        start = int((len(gyro_pitch) - n_seconds_max * sample_rate) / 2)
        end = int((len(gyro_pitch) + n_seconds_max * sample_rate) / 2)
        sig = gyro_pitch[start:end]
    else:
        sig = gyro_pitch

    # Apply FFT to the signal
    fft_result = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), d=1 / sample_rate)

    # Consider only the positive frequencies (real signal)
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = np.abs(fft_result[:len(fft_result) // 2])

    # Find the frequency with the highest magnitude
    dominant_frequency = positive_freqs[np.argmax(positive_magnitude)]

    # Return cadence in Hz
    return dominant_frequency


def get_zero_crossings(signal: np.ndarray, direction: str = 'positive'):
    val = 2 if direction == 'positive' else -2
    sign = np.sign(signal)
    zc = np.where(np.diff(sign) == val)[0] + 1
    return zc


def get_foot_events_running(t_ms: np.ndarray, acc: np.ndarray, gyr: np.ndarray, side: str):
    """
    Get the gait events for a single foot mounted IMU
    :param t_ms: timestamp in ms
    :param acc: 3D accelerometer signal.
    :param gyr: 3D gyroscope signal.
    :param side: "left" or "right" foot
    :return: Dictionary containing lists of initial and terminal contact events. index based
    """
    events = dict()
    sample_rate = sample_rate_from_timestamp(t_ms)
    if sample_rate < 100:
        sample_rate *= 1000

    # Calculate resultant acceleration and jerk for later use
    sos_lp = sg.butter(4, 30, 'lowpass', fs=sample_rate, output='sos')
    acc_filt = sg.sosfiltfilt(sos_lp, acc, axis=0)
    acc_res = np.linalg.norm(acc_filt, axis=1)

    jerk_res = np.concatenate((np.zeros(1), np.diff(acc_res)))  # *SAMPLE_RATE
    acc_vert = acc[:, 2]  # simplification in lack of functional calibration

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Adapted from: Falbriard et al. 2018:
    # Accurate Estimation of Running Temporal Parameters Using Foot-Worn Inertial Sensors
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Get cadence estimate from pitch gyro signal
    gyro_pitch = gyr[:, 0]
    if 'r' in side.lower():  # simplified mirroring of the right foot pitch gyro signal
        gyro_pitch *= -1
    gyro_pitch_filt = sg.sosfiltfilt(sos_lp, gyro_pitch)
    # todo: get_cadence_estimate seems unreliable when running overground
    cadence_est = np.min((2, get_cadence_estimate(gyro_pitch_filt, sample_rate)))  # plausibility check...
    # print(f'{cadence_est=}')
    # Heavily filter the pitch signal of the foot gyro to get the mid swing events
    sos_lp_mid_swing = sg.butter(4, 2, 'lowpass', fs=sample_rate, output='sos')
    gyro_pitch_filt_mid_swing = sg.sosfiltfilt(sos_lp_mid_swing, gyro_pitch)

    # get the peaks of that signal as the mid swing events
    min_height = np.quantile(gyro_pitch_filt_mid_swing, 0.66)
    mid_swing, _ = sg.find_peaks(gyro_pitch_filt_mid_swing,
                                 distance=(0.6 * (sample_rate / cadence_est)),
                                 height=min_height)
    ic_list = []
    tc_list = []
    # loop over mid swing events
    for s in range(len(mid_swing) - 1):
        start = mid_swing[s]
        end = mid_swing[s + 1]
        if end - start > (2 * sample_rate / cadence_est):
            # happens when stopping, standing and starting again or switching to walking
            continue
        # get signal slices for this midswing-midswing phase
        gyr_slice = gyro_pitch_filt[start:end]
        jerk_slice = jerk_res[start:end]
        acc_vert_slice = acc_vert[start:end]
        t_slice = t_ms[start:end]
        #
        # InitialContact
        #
        # get the zero crossings
        zero_crossings = np.where(np.diff(np.sign(jerk_slice)))[0]
        # get the max of the jerk (must be in the first half of the slice, but after the first zero-crossing)
        if len(zero_crossings) == 0:
            warnings.warn('No zero crossing found. Skipping...')
            continue
        zc_0 = zero_crossings[0]
        jerk_max = np.argmax(jerk_slice[zc_0:int(len(jerk_slice) / 2)]) + zc_0
        # jerk_max = np.argmax(jerk_slice)
        # get the gyro pitch minimum during the first third of mid-swing to mid-swing
        # (adjusted k1 in Falbriard et al. 2018 paper), i.e. ic
        ic_slice = None
        tc_slice = None
        third = int(len(gyr_slice) / 3)
        try:
            # ic_slice = zero_crossings[np.where(zero_crossings < jerk_max)[0][-1]]
            ic_slice = np.argmin(gyr_slice[:third])
            # refinement for fore-foot striker: there might be a double peak. But we only want the first one
            pks = sg.find_peaks(-gyr_slice[:ic_slice], height=-0.66 * gyr_slice[ic_slice])[0]
            if len(pks) > 0:
                ic_slice = pks[-1]
        except Exception as e:
            print(e)
        if ic_slice is None:
            continue
        #
        # TerminalContact
        #
        # find the minimum of the gyro pitch in the second two-thirds of mid-swing to mid-swing
        # (adjusted t1 in Falbriard et al. 2018 paper), i.e. tc
        try:
            tc_slice = np.argmin(gyr_slice[third:]) + third
            # todo: check how we can make this more robust
        except Exception as e:
            print(e)
        if tc_slice is None:
            continue
        ic_list.append(start + ic_slice)
        tc_list.append(start + tc_slice)

    events['ic'] = ic_list
    events['tc'] = tc_list
    return events


def get_tibia_events_running_sinclair(t: np.ndarray, acc: np.ndarray, gyr: np.ndarray, side: str):
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Adapted from: Sinclair et al. 2013:
    # Determination of Gait Events Using an Externally Mounted Shank Accelerometer
    # # # # # # # # # # # # # # # # # # # # # # # #
    events = dict()
    acc_axial = -acc[:, 0]
    sample_rate = sample_rate_from_timestamp(t)
    sos_lp = sg.butter(4, 60, 'lowpass', fs=sample_rate, output='sos')
    acc_axial_filt = sg.sosfiltfilt(sos_lp, acc_axial)

    max_stride_rate = 110  # strides per minute todo: replace with a robust cadence estimate
    t_dist_min = int(sample_rate / max_stride_rate * 60)
    min_height = np.quantile(acc_axial_filt, 0.95)
    peaks, _ = sg.find_peaks(acc_axial_filt, distance=t_dist_min, height=min_height)
    # 50 ms window for robust zero-crossing detection (original article says 20, but that doesn't work...)
    window_start = int(60 * 1000 / sample_rate)
    window_end = int(10 * 1000 / sample_rate)
    zc = get_zero_crossings(acc_axial)
    ic_list = []
    tc_list = []
    # find the zero-crossings before the peaks
    for peak in peaks:
        if (peak - window_start) < 0:  # edge case (first discovered ic)
            continue
        # # # # # # # # # # # #
        # GET CLOSEST Zero-Crossing (within window)
        # # # # # # # # # # # #
        # First requirement: IC must be before the peak
        ic1 = np.delete(zc, np.where((zc - peak) > 0))
        # Second requirement: IC must be within a certain window before the peak
        ic1 = np.delete(ic1, np.where((ic1 - peak) < -window_start))
        if len(ic1) == 0:
            ic1 = None
        else:
            ic1 = ic1[-1]
        # # # # # # # # # # # #
        # GET CLOSEST LOCAL MINIMUM (within window)
        # # # # # # # # # # # #
        temp_ic2, _ = sg.find_peaks(-acc_axial[peak - window_start:peak - window_end])
        # must be at least 10 ms before the peak
        if len(temp_ic2) == 0:
            ic2 = None
        else:
            ic2 = peak - window_start + temp_ic2[-1]
        # choose the one that's closer to the peak
        if ic1 is None and ic2 is None:
            continue
        elif ic1 is None:
            ic = ic2
        elif ic2 is None:
            ic = ic1
        else:
            ic = max((ic1, ic2))
        ic_list.append(ic)
    ic_array = np.array(ic_list)
    events['ic'] = ic_list
    events['tc'] = tc_list
    return events


def get_tibial_shock(t: np.ndarray, acc: np.ndarray, gyr: np.ndarray, side: str, path_plot: Path | None = None):
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Refined version of step detection for running at the tibia using the gyroscope AND accelerometer signal
    # # # # # # # # # # # # # # # # # # # # # # # #
    events = dict()
    make_plot = path_plot is not None
    sample_rate = sample_rate_from_timestamp(t)
    sos_lp = sg.butter(4, 30, 'lowpass', fs=sample_rate, output='sos')

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Adapted from: Falbriard et al. 2018:
    # Accurate Estimation of Running Temporal Parameters Using Foot-Worn Inertial Sensors
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Get cadence estimate from pitch gyro signal
    gyro_pitch = gyr[:, 1]
    if 'r' in side.lower():  # simplified mirroring of the right foot pitch gyro signal
        gyro_pitch *= -1
    gyro_pitch_filt = sg.sosfiltfilt(sos_lp, gyro_pitch)

    # todo: get_cadence_estimate seems unreliable when running overground
    cadence_est = np.min((2, get_cadence_estimate(gyro_pitch_filt, sample_rate)))  # plausibility check...
    # print(f'{cadence_est=}')
    # Heavily filter the pitch signal of the foot gyro to get the mid swing events
    sos_lp_mid_swing = sg.butter(4, 2, 'lowpass', fs=sample_rate, output='sos')
    gyro_pitch_filt_mid_swing = sg.sosfiltfilt(sos_lp_mid_swing, gyro_pitch)

    # get the peaks of that signal as the mid swing events
    min_height = np.quantile(gyro_pitch_filt_mid_swing, 0.66)
    mid_swing, _ = sg.find_peaks(gyro_pitch_filt_mid_swing,
                                 distance=(0.6 * (sample_rate / cadence_est)),
                                 height=min_height)
    tibial_shock = []
    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(24, 10))
        ax2 = ax.twinx()

    for start, end in zip(mid_swing, mid_swing[1:]):
        acc_axial = -acc[start:end, 0]
        min_height = np.quantile(acc_axial, 0.95)
        zc = get_zero_crossings(gyro_pitch_filt[start:end], direction='negative')[0]
        peaks, shocks = sg.find_peaks(acc_axial, height=min_height)
        # filter peaks which are before the gyro pitch zero crossing
        peaks = peaks[peaks > zc]
        peak = peaks[0]
        shock = shocks['peak_heights'][0]
        tibial_shock.append(shock)
        if make_plot:
            ax.plot(gyro_pitch_filt[start:end], color='b', linewidth=0.2)
            ax2.plot(acc_axial, color='k', linewidth=0.2)
            ax2.scatter(peak, acc_axial[peak], color='r')

    if make_plot:
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title(path_plot.stem)
        fig.tight_layout()
        fig.savefig(path_plot)

    return tibial_shock


def get_sacrum_events_running(t: np.ndarray, acc: np.ndarray, gyr: np.ndarray, side: None):
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Adapted from: Day et al. 2021:
    # Low-pass filter cutoff frequency affects sacral-mounted inertial measurement unit estimations of peak vertical
    # ground reaction force and contact time during treadmill running
    # # # # # # # # # # # # # # # # # # # # # # # #
    events = dict()
    events['left'] = dict()
    events['right'] = dict()
    acc_vert = -acc[:, 0]
    gyr_ap = gyr[:, 2]  # KiNetBlue z-axis pointing back from sacrum -> positive values mean left obliquity motion

    cutoff_freq = 8
    sample_rate = sample_rate_from_timestamp(t)
    sos_lp = sg.butter(4, cutoff_freq, 'lowpass', fs=sample_rate, output='sos')
    acc_vert_filt = sg.sosfiltfilt(sos_lp, acc_vert)
    ic_array = get_zero_crossings(acc_vert_filt)
    # clean up implausible close zero-crossings:
    avg_samples_per_step = np.median(np.diff(ic_array)).astype(int)
    delete = np.where(np.diff(ic_array) < (0.5 * avg_samples_per_step))
    ic_array = np.delete(ic_array, delete)

    # now we need to find out which side the ics are associated with
    # easily achievable through antero-posterior gyro signal peaks
    ic_1 = ic_array[::2]
    ic_2 = ic_array[1::2]

    def get_sign(ic_ipsi, ic_contra):
        sig = 0
        initial_peaks = []
        while ic_ipsi[0] > ic_contra[0]:
            ic_contra = ic_contra[1:]
        for i1, i2 in zip(ic_ipsi, ic_contra):
            step_cycle = gyr_ap[i1:i2]
            window = np.max((50, int(0.5 * (i2 - i1))))
            next_peaks, _ = sg.find_peaks(abs(gyr_ap[i1:i1 + window]))
            if len(next_peaks) > 0:
                ind = np.argmax(abs(step_cycle[next_peaks]))
                next_peak = next_peaks[ind] + i1
                sig += np.sign(gyr_ap[next_peak])
                initial_peaks.append(next_peak)
        return sig, initial_peaks

    sign_1, ip1 = get_sign(ic_1, ic_2)
    ic_right = ic_1 if sign_1 > 0 else ic_2
    sign_2, ip2 = get_sign(ic_2, ic_1)
    ic_left = ic_1 if sign_2 > 0 else ic_2
    if np.array_equal(ic_left, ic_right):
        raise ValueError('Something went wrong with the foot matching')

    plt.plot(acc_vert_filt)
    plt.plot(acc_vert_filt * 10)
    plt.plot(gyr[:, 2])
    for ic in ic_array:
        plt.axvline(ic, color='k')
    plt.axhline(0, color='k')
    plt.scatter(ip1, gyr_ap[ip1], color='r')
    plt.scatter(ip2, gyr_ap[ip2], color='g')
    events['left']['ic'] = ic_left
    events['right']['ic'] = ic_right
    return events


def get_chest_events_running(t: np.ndarray, acc: np.ndarray, gyr: np.ndarray, side: None):
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Adapted from: Day et al. 2021:
    # Low-pass filter cutoff frequency affects sacral-mounted inertial measurement unit estimations of peak vertical
    # ground reaction force and contact time during treadmill running
    # # # # # # # # # # # # # # # # # # # # # # # #
    acc_corr, gyr_corr = reorient_to_gravity_treadmill_running(acc, gyr)
    events = dict()
    events['left'] = dict()
    events['right'] = dict()
    acc_vert = acc_corr[:, 2]
    gyr_ap = -gyr_corr[:, 0]  # KiNetBlue z-axis pointing back from sacrum -> positive values mean left obliquity motion
    gyr_vert = gyr_corr[:,
               2]  # KiNetBlue z-axis pointing back from sacrum -> positive values mean left obliquity motion
    sample_rate = sample_rate_from_timestamp(t)
    sos_lp = sg.butter(4, 5, 'lowpass', fs=sample_rate, output='sos')
    acc_vert_filt = sg.sosfiltfilt(sos_lp, acc_vert)
    ic_array = get_zero_crossings(acc_vert_filt - 0.5)  # -0.5 for robustness. i.e. 1g threshold
    # clean up implausible close zero-crossings:
    avg_samples_per_step = np.median(np.diff(ic_array)).astype(int)

    delete = np.where(np.diff(ic_array) < (0.55 * avg_samples_per_step))[0] + 1
    delete_final = delete.copy()
    for dele in delete:
        d = ic_array[dele]
        window_next = [int(0.9 * avg_samples_per_step), int(1.1 * avg_samples_per_step)] + d
        window_prev = d - [int(1.1 * avg_samples_per_step), int(0.9 * avg_samples_per_step)]
        if any(np.where((window_next[0] < ic_array) & (ic_array < window_next[1]))) and \
                any(np.where((window_prev[0] < ic_array) & (ic_array < window_prev[1]))):
            delete_final = np.delete(delete_final, np.where(delete_final == dele))
    ic_array = np.delete(ic_array, delete_final)
    # now we need to find out which side the ics are associated with
    # easily achievable through antero-posterior gyro signal peaks
    ic_1 = ic_array[::2]
    ic_2 = ic_array[1::2]

    def get_sign(ic_ipsi, ic_contra, gyr_vert):
        sig = 0
        while ic_ipsi[0] > ic_contra[0]:
            ic_contra = ic_contra[1:]
        for i1, i2 in zip(ic_ipsi, ic_contra):
            step_cycle = gyr_vert[i1:i2]
            # plt.plot(step_cycle, 'k')
            sig += np.trapz(step_cycle)
        return np.sign(sig)

    sign_1 = get_sign(ic_1, ic_2, gyr_vert)
    ic_right = ic_1 if sign_1 > 0 else ic_2
    sign_2 = get_sign(ic_2, ic_1, gyr_vert)
    ic_left = ic_1 if sign_2 > 0 else ic_2
    if np.array_equal(ic_left, ic_right):
        raise ValueError('Something went wrong with the foot matching')
    events['left']['ic'] = ic_left
    events['right']['ic'] = ic_right
    return events


def reorient_to_gravity_treadmill_running(acc: np.ndarray, gyr: np.ndarray):
    # align imu to gravity. Assumption: average of accelerometer axes form gravity vector
    g_vec = np.mean(acc, axis=0)
    g_vec = g_vec / np.linalg.norm(g_vec)
    ml_vector = np.array([0, 1, 0])

    corr_mat = get_correction_matrix_from_axis_axis(ml_vector, g_vec)
    acc_corr = np.matmul(corr_mat, acc.T).T
    gyr_corr = np.matmul(corr_mat, gyr.T).T

    return acc_corr, gyr_corr


def get_correction_matrix_from_axis_axis(knee_axis: np.ndarray, gravity_axis: np.ndarray, plot: bool = False):
    # calculate correction matrix to have new axis:
    # TODO: Verify, that AHRS expects it like that!
    # x: Anteroposterior - pointing anterior
    # y: Mediolateral - pointing medial
    # z: Caudocranial - pointing up
    vertical_axis = gravity_axis  # new z-axis

    ap_axis = np.cross(knee_axis, vertical_axis)  # new x-axis
    ml_axis = np.cross(vertical_axis, ap_axis)  # new y-axis
    corr_mat33 = np.array((ap_axis, ml_axis, vertical_axis)).T

    gravity_axis_new = np.matmul(gravity_axis, corr_mat33)
    knee_axis_new = np.matmul(knee_axis, corr_mat33)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(0, 0, 0, 1, 0, 0, color='r')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b')

        ax.quiver(0, 0, 0, ap_axis[0], ap_axis[1], ap_axis[2], color='r', linestyle='--')
        ax.quiver(0, 0, 0, ml_axis[0], ml_axis[1], ml_axis[2], color='g', linestyle='--')
        ax.quiver(0, 0, 0, vertical_axis[0], vertical_axis[1], vertical_axis[2], color='b', linestyle='--')

        ax.quiver(0, 0, 0, gravity_axis[0], gravity_axis[1], gravity_axis[2], color='orange')
        ax.quiver(0, 0, 0, knee_axis[0], knee_axis[1], knee_axis[2], color='grey')

        ax.quiver(0, 0, 0,
                  gravity_axis_new[0], gravity_axis_new[1], gravity_axis_new[2],
                  color='orange', linestyle='--')
        ax.quiver(0, 0, 0,
                  knee_axis_new[0], knee_axis_new[1], knee_axis_new[2],
                  color='grey', linestyle='--')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()

    return np.matmul(corr_mat33, np.array([[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, 1]]))
    return corr_mat33


# Register detectors in a dictionary for easy access
detectors: dict[str, callable] = {
    "foot": get_foot_events_running,
    "tibia": get_tibia_events_running_sinclair,
    "pelvis": get_sacrum_events_running,
    "trunk": get_chest_events_running,
}


def get_running_events(t: np.ndarray,
                       acc: np.ndarray,
                       gyr: np.ndarray,
                       sensor_location: str,
                       side: str | None = None,
                       ):
    return detectors[sensor_location](t, acc, gyr, side)
    # match sensor_location:
    #     case "foot".LEFT_FOOT | SensorLocation.RIGHT_FOOT:
    #         return foot_events_running(t_ms=t, acc=acc, gyr=gyr, side=side)
    #     case SensorLocation.LEFT_SHANK | SensorLocation.RIGHT_SHANK:
    #         return tibia_events_running_sinclair(t=t, acc=acc, gyr=gyr, side=side)
    #     case SensorLocation.PELVIS:
    #         return sacrum_events_running(t=t, acc=acc, gyr=gyr)
    #     case SensorLocation.MID_TRUNK | SensorLocation.UPPER_TRUNK:
    #         return chest_events_running(t=t, acc=acc, gyr=gyr)
