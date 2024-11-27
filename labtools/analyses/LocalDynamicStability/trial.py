from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LDSTrial:
    participant_id: str
    trial_id: str
    path_data: Path
    sensor_location: str
    # private
    _path_data_file = None
    _data = None

    def __post_init__(self):
        self._validate_input()
        self._load_data()

    def _validate_input(self):
        self._path_data_file = self.path_data.joinpath(f'{self.sensor_location}.csv')
        if not self._path_data_file.exists():
            raise FileNotFoundError(f'The file {self._path_data_file.name} does not exist for participant {self.participant_id} trial {self.trial_id}.')

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        return self._load_data()

    def _load_data(self) -> np.ndarray:
        self._data = np.genfromtxt(self._path_data_file, delimiter=',', skip_header=1)
        return self._data
