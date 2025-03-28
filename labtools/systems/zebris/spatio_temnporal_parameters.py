from pathlib import Path

import numpy as np

from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d


def steprate_from_events(events: dict, sample_rate: int) -> tuple[float, float]:
    """
    Calculate step rate from force events
    :param events: dict of force events
    :param sample_rate: sample rate of the force signal
    :return: average steps per minute, standard deviation of steps per minute
    """
    ic_time_s = events['ic'] * 1 / sample_rate
    step_duration = np.diff(ic_time_s)
    step_rate_per_step = 60 / step_duration
    steps_per_minute_avg = step_rate_per_step.mean()
    steps_per_minute_std = step_rate_per_step.std()
    return steps_per_minute_avg, steps_per_minute_std


def contact_time_from_events(events: dict, sample_rate: int) -> tuple[float, float]:
    """
    Calculate contact time from force events
    :param events: dict of force events
    :param sample_rate: sample rate of the force signal
    :return: average contact time, standard deviation of contact times in seconds
    """
    ic_time_s = events['ic'] * 1 / sample_rate
    tc_time_s = events['tc'] * 1 / sample_rate
    contact_times = tc_time_s - ic_time_s
    contact_time_avg = contact_times.mean()
    contact_time_std = contact_times.std()
    return contact_time_avg, contact_time_std


def flight_time_from_events(events: dict, sample_rate: int) -> tuple[float, float]:
    """
    Calculate flight time from force events
    :param events: dict of force events
    :param sample_rate: sample rate of the force signal
    :return: average flight time, standard deviation of flight times in seconds
    """
    ic_time_s = events['ic'] * 1 / sample_rate
    tc_time_s = events['tc'] * 1 / sample_rate
    flight_times = ic_time_s[1:] - tc_time_s[:-1]
    flight_time_avg = flight_times.mean()
    flight_time_std = flight_times.std()
    return flight_time_avg, flight_time_std


def normalized_ground_contact_time_from_events(events: dict) -> tuple[float, float]:
    """
    Calculate normalized ground contact time from force events
    :param events: dict of force events
    :return: average normalized ground contact time, standard deviation of normalized ground contact times
    """
    ics = np.array(events['ic'])
    tcs = np.array(events['tc'])
    ngcts = np.array([])
    for side in [0, 1]:
        contact_frames = tcs[side::2] - ics[side::2]
        stride_frames = np.diff(ics[side::2])
        length = min(len(contact_frames), len(stride_frames))
        ngcts = np.append(ngcts, contact_frames[:length] / stride_frames[:length])
    normalized_contact_time_avg = ngcts.mean()
    normalized_contact_time_std = ngcts.std()
    return normalized_contact_time_avg, normalized_contact_time_std

def analyze(file: Path, include_std: bool = False):
    data, meta = load_c3d(file)
    sample_rate = data['analog_rate']
    f_z = get_force(data)
    evt = get_force_events_treadmill(f_z=f_z,
                                     sample_rate=sample_rate)

    steprate_avg, steprate_std = steprate_from_events(evt, sample_rate)
    contact_time_avg, contact_time_std = contact_time_from_events(evt, sample_rate)
    flight_time_avg, flight_time_std = flight_time_from_events(evt, sample_rate)
    ngct_avg, ngct_std = normalized_ground_contact_time_from_events(evt)

    out = {
        'steps_per_minute': round(steprate_avg, 1),
        'contact_time_ms': int(contact_time_avg * 1000),
        'flight_time_ms': int(flight_time_avg * 1000),
        'normalized_ground_contact_time': round(ngct_avg, 3)
    }
    if include_std:
        out['steps_per_minute_std'] = round(steprate_std, 1)
        out['contact_time_ms_std'] = int(contact_time_std * 1000)
        out['flight_time_ms_std'] = int(flight_time_std * 1000)
        out['normalized_ground_contact_time_std'] = round(ngct_std, 3)

    return out


if __name__ == '__main__':
    path = Path("/Users/dominikfohrmann/dev/github/BiomechLabTools/labtools/test/data/c3d_testfile_zebris.c3d")
    out = analyze(path)
    print(out)
