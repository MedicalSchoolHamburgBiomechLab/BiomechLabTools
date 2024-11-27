from pathlib import Path

import c3d
import numpy as np


def load_c3d(file_path: str | Path):
    with open(file_path, 'rb') as handle:
        reader = c3d.Reader(handle)
        try:
            point_labels = reader.point_labels
            point_labels = [pl.replace(" ", "") for pl in point_labels]
        except Exception as err:
            point_labels = []
        data = dict()
        meta = dict()

        try:
            for key, values in reader.groups.items():
                if not isinstance(key, str):
                    continue
                meta[key] = dict()
                if not isinstance(values, c3d.Group):
                    continue
                for k, v in values.params.items():
                    if not isinstance(v, c3d.Param):
                        continue
                    meta[key][k] = v
        except Exception as err:
            # print(err)
            pass

        data['marker'] = dict.fromkeys(point_labels)
        data['analog'] = None
        data['analog_rate'] = reader.analog_rate
        data['point_rate'] = reader.point_rate
        try:
            creation_date = reader.get('PROCESSING').get_string('CREATION DATE')  # is padded to 255 characters
            data['creation_date'] = creation_date.strip()
            creation_time = reader.get('PROCESSING').get_string('CREATION TIME')  # is padded to 255 characters
            data['creation_time'] = creation_time.strip()
        except Exception as err:
            # print(err)
            pass
        # ratio = int(reader.analog_rate/reader.point_rate)
        for i, points, analog in reader.read_frames():
            for m, marker in enumerate(point_labels):
                data['marker'][marker] = points[m, :3] if np.all(data['marker'][marker] is None) \
                    else np.vstack((data['marker'][marker], points[m, :3]))
            if np.all(data['analog'] is None):
                data['analog'] = analog.T
            else:
                data['analog'] = np.vstack((data['analog'], analog.T))

    return data, meta
