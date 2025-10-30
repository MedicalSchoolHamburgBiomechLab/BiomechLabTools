from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from labtools.analyses.LocalDynamicStability.trial import LDSTrial
from labtools.utils.file_handling import get_folder_list


@dataclass
class Participant:
    participant_id: str
    path_data: Path
    sensor_location: str
    # private
    _trials: List[LDSTrial] = field(default_factory=list)  # Use default_factory for mutable default

    def __post_init__(self):
        self._validate_input()
        self._init_trials()

    def _validate_input(self):
        if not isinstance(self.participant_id, str):
            raise TypeError("The participant ID must be a string.")
        if not isinstance(self.path_data, Path):
            raise TypeError("The path to the data must be a Path object.")
        if not self.path_data.exists():
            raise FileNotFoundError(f"The path {self.path_data} does not exist.")

    def _init_trials(self):
        trial_ids = get_folder_list(self.path_data)
        for trial_id in trial_ids:
            try:
                trial = LDSTrial(participant_id=self.participant_id,
                                 trial_id=trial_id,
                                 path_data=self.path_data.joinpath(trial_id),
                                 sensor_location=self.sensor_location)
            except FileNotFoundError as e:
                print(f'Trial does not exist: {trial_id}. {e}')
                continue
            except Exception as e:
                print(f'Error creating trial {trial_id} for participant {self.participant_id}: {e}')
                continue
            self._trials.append(trial)

    @property
    def trials(self) -> List[LDSTrial]:
        return self._trials
