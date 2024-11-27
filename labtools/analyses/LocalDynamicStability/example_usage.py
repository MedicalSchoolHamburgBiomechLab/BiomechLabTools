import os
from pathlib import Path

from dotenv import load_dotenv

from analysis import LDSAnalysis

load_dotenv('.env')

if __name__ == '__main__':
    # Define participants, conditions, sensor locations, and root path
    PATH_DATA_ROOT = os.environ.get('PATH_DATA_ROOT')
    if PATH_DATA_ROOT is None:
        raise ValueError("The PATH_DATA_ROOT environment variable must be set.")
    path_data_in = Path(PATH_DATA_ROOT).joinpath('running stability')
    path_data_out = Path(PATH_DATA_ROOT).joinpath('analysis')
    # data must be structured as follows:
    # data_path
    # ├── P001
    # │   ├── PRE
    # │   │   ├── pelvis.csv
    # │   │   ├── thigh_left.csv
    # │   │   ├── ...
    # │   ├── POST
    # │   │   ├── pelvis.csv
    # │   │   ├── ...
    # ├── P002
    # │   ├── ...

    # sensor_locations = ['foot_right']
    sensor_locations = ['tibia_right', 'foot_right']
    for sensor_location in sensor_locations:
        analysis = LDSAnalysis(path_data_in=path_data_in,
                               path_data_out=path_data_out,
                               sensor_location=sensor_location,
                               force_recalculate=True)
        analysis.summary()
        # analysis.compute_time_delays()
        analysis.time_delay_summary()
        # analysis.compute_embedding_dimensions()
        analysis.embedding_dimension_summary()
        # analysis.compute_divergence_curves()

        # plot_divergence_curves(analysis.divergence_curves,path_data_out, sensor_location)

        # analysis.set_fit_interval(end=20)
        # analysis.compute_divergence_exponents()
        # print(analysis.divergence_exponents)
