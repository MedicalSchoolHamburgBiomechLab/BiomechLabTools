from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d


def analyze(file: Path):
    data, meta = load_c3d(file)
    sample_rate = data['analog_rate']
    f_z = get_force(data)
    plt.plot(f_z)
    evt = get_force_events_treadmill(f_z=f_z,
                                     sample_rate=sample_rate)
    t = np.arange(0, len(f_z)) * 1 / sample_rate
    ic_time = t[evt['ic']]
    tc_time = t[evt['tc']]
    step_duration = np.diff(ic_time)
    steps_per_minute_avg = 60 / step_duration.mean()
    contact_times = tc_time - ic_time
    contact_time_avg = contact_times.mean()
    flight_times = ic_time[1:] - tc_time[:-1]
    flight_time_avg = flight_times.mean()
    return {
        'steps_per_minute_avg': int(steps_per_minute_avg),
        'contact_time_avg': contact_time_avg.round(3),
        'flight_time_avg': flight_time_avg.round(3)
    }
