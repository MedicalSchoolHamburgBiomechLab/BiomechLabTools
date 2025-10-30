import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from labtools.analyses.LocalDynamicStability.algorithms import rosenstein_divergence
from labtools.analyses.LocalDynamicStability.participant import Participant
from labtools.signal_analysis.non_linear.delay_coordinate_embedding import state_space_reconstruction
from labtools.signal_analysis.non_linear.false_nearest_neighbours import false_nearest_neighbours
from labtools.signal_analysis.non_linear.mutual_information import minimum_average_mutual_information
from labtools.utils.convenience import DEBUG, process_on_dataframe
from labtools.utils.file_handling import get_participant_folder_list, load_dataframe, save_dataframe


def calculate_divergence_exponent(divergence_curve: np.ndarray, fit_interval: tuple):
    x = np.arange(fit_interval[0], fit_interval[1])
    y = divergence_curve[fit_interval[0]:fit_interval[1]]
    beta = np.polyfit(x, y, 1)
    return beta[0] * len(divergence_curve)


@dataclass
class LDSAnalysis:
    path_data_in: Path
    path_data_out: Path
    sensor_location: str
    force_recalculate: bool = False
    # private
    _participants: List[Participant] = field(default_factory=list)
    _participant_ids: List[str] = field(default_factory=list)
    _conditions: List[str] = field(default_factory=list)
    _signals: pd.DataFrame = None  # (3x10,000) for each participant and condition
    _time_delays: pd.DataFrame = None  # (x, y, z) time delays for each participant and condition
    _tau: int = None
    _embedding_dimensions: pd.DataFrame = None  # (x, y, z) time delays for each participant and condition
    _dE: int = None
    _state_spaces: pd.DataFrame = None
    _divergence_curves: pd.DataFrame = None  # (101x1) divergence curves for each participant and condition
    _fit_interval: tuple[int, int] = None  # (start, end) for linear fit of divergence curves for divergence exponent
    _divergence_exponents: pd.DataFrame = None  # (1x1) divergence exponents for each participant and condition

    def __post_init__(self):
        self._validate_input()
        self._set_paths()
        self._init_data()
        if self.force_recalculate or self._signals is None:
            self._init_from_environment()
        else:
            self._init_from_signals()

    def _validate_input(self):
        if not isinstance(self.path_data_in, Path):
            raise TypeError("The path to the input data root must be a Path object.")
        if not isinstance(self.path_data_out, Path):
            raise TypeError("The path to the output data must be a Path object.")
        if not self.path_data_in.exists():
            raise FileNotFoundError(f"The path {self.path_data_in} does not exist.")
        if not isinstance(self.sensor_location, str):
            raise TypeError("The sensor location must be a string.")

    def _set_paths(self):
        self._path_signals = self.path_data_out.joinpath(f'signals_{self.sensor_location}.pkl')
        self._path_time_delays = self.path_data_out.joinpath(f'time_delays_{self.sensor_location}.json')
        self._path_embedding_dimensions = self.path_data_out.joinpath(f'embedding_dimensions_{self.sensor_location}.json')
        self._path_state_spaces = self.path_data_out.joinpath(f'state_spaces_{self.sensor_location}.pkl')
        self._path_divergence_curves = self.path_data_out.joinpath(f'divergence_curves_{self.sensor_location}.pkl')
        self._path_divergence_exponents = self.path_data_out.joinpath(f'divergence_exponents_{self.sensor_location}.xlsx')

    def _init_from_environment(self):
        self._participant_ids = get_participant_folder_list(self.path_data_in)
        for p_id in self._participant_ids:
            try:
                participant = Participant(participant_id=p_id,
                                          sensor_location=self.sensor_location,
                                          path_data=self.path_data_in.joinpath(p_id))
            except FileNotFoundError as e:
                print(f'Participant does not exist: {p_id}. {e}')
                continue
            self._participants.append(participant)

    def _init_from_signals(self):
        self._signals = self._load_signals()
        self._participant_ids = self._signals.index.tolist()
        self._conditions = self._signals.columns.tolist()

    def _init_data(self):
        if self._path_signals.exists():
            self._signals = self._load_signals()
        if self._path_time_delays.exists():
            self._time_delays = load_dataframe(path_filename=self._path_time_delays)
        if self._path_embedding_dimensions.exists():
            self._embedding_dimensions = load_dataframe(path_filename=self._path_embedding_dimensions)
        if self._path_state_spaces.exists():
            self._state_spaces = load_dataframe(path_filename=self._path_state_spaces)
        if self._path_divergence_curves.exists():
            self._divergence_curves = load_dataframe(path_filename=self._path_divergence_curves)
        if self._path_divergence_exponents.exists():
            self._divergence_exponents = load_dataframe(path_filename=self._path_divergence_exponents)
        # variables:
        if self._time_delays is not None:
            self._calculate_time_delay()  # sets self._tau
        if self._embedding_dimensions is not None:
            self._calculate_embedding_dimension()  # sets self._dE

    @property
    def participants(self) -> List[Participant]:
        return self._participants

    @property
    def participant_ids(self) -> List[str]:
        if self._participants:
            return [p.participant_id for p in self._participants]
        return self._participant_ids

    @property
    def conditions(self) -> List[str]:
        if not self._conditions:
            unique_conditions = set()
            for participant in self._participants:
                for trial in participant.trials:
                    unique_conditions.add(trial.trial_id)
            self._conditions = list(unique_conditions)
            self._conditions.sort()
        return self._conditions

    def summary(self):
        print('#' * 80)
        print(f"Running stability analysis for {self.sensor_location}")
        print(f'Participants: {self.participant_ids}')
        print(f'Conditions: {self.conditions}')
        print(f'Force recalculate: {self.force_recalculate}')
        print(f'Path data in: {self.path_data_in}')
        print(f'Path data out: {self.path_data_out}')
        if self._tau is not None:
            print(f'Time delay: {self._tau}')
        if self._dE is not None:
            print(f'Embedding dimension: {self._dE}')
        print('#' * 80)

    @property
    def signals(self) -> pd.DataFrame | None:
        if self._signals is not None:
            return self._signals
        if self._path_signals.exists():
            self._signals = load_dataframe(path_filename=self._path_signals)
            return self._signals
        warnings.warn('Signals not computed yet.')
        return None

    @property
    def time_delays(self) -> pd.DataFrame | None:
        if self._time_delays is not None:
            return self._time_delays
        if self._path_time_delays.exists():
            self._time_delays = load_dataframe(path_filename=self._path_time_delays)
            return self._time_delays
        warnings.warn('Time delays not computed yet.')
        return None

    @property
    def embedding_dimensions(self) -> pd.DataFrame | None:
        if self._embedding_dimensions is not None:
            return self._embedding_dimensions
        if self._path_embedding_dimensions.exists():
            self._embedding_dimensions = load_dataframe(path_filename=self._path_embedding_dimensions)
            return self._embedding_dimensions
        warnings.warn('Embedding dimensions not computed yet.')
        return None

    @property
    def state_spaces(self) -> pd.DataFrame | None:
        if self._state_spaces is not None:
            return self._state_spaces
        if self._path_state_spaces.exists():
            self._state_spaces = load_dataframe(path_filename=self._path_state_spaces)
            return self._state_spaces
        warnings.warn('State spaces not computed yet.')
        return None

    @property
    def divergence_curves(self) -> pd.DataFrame | None:
        if self._divergence_curves is not None:
            return self._divergence_curves
        if self._path_divergence_curves.exists():
            self._divergence_curves = load_dataframe(path_filename=self._path_divergence_curves)
            return self._divergence_curves
        warnings.warn('Divergence curves not computed yet.')
        return None

    @property
    def fit_interval(self) -> tuple[int, int]:
        return self._fit_interval

    @property
    def divergence_exponents(self) -> pd.DataFrame | None:
        if self._divergence_exponents is not None:
            return self._divergence_exponents
        if self._path_divergence_exponents.exists():
            self._divergence_exponents = load_dataframe(path_filename=self._path_divergence_exponents)
            return self._divergence_exponents
        warnings.warn('Divergence exponents not computed yet.')
        return self._divergence_exponents

    @property
    def time_delay(self) -> int:
        if self._tau is not None:
            return self._tau
        self._calculate_time_delay()
        if self._tau is None:
            raise ValueError('Time delay could not be calculated.')
        return self._tau

    def set_time_delay(self, tau: int):
        self._tau = tau

    def _calculate_time_delay(self):
        df_time_delays = self._time_delays
        x_values = df_time_delays.apply(lambda col: col.map(lambda d: d.get('x') if isinstance(d, dict) else np.nan))
        y_values = df_time_delays.apply(lambda col: col.map(lambda d: d.get('y') if isinstance(d, dict) else np.nan))
        z_values = df_time_delays.apply(lambda col: col.map(lambda d: d.get('z') if isinstance(d, dict) else np.nan))

        x_mean = x_values.stack().mean()
        y_mean = y_values.stack().mean()
        z_mean = z_values.stack().mean()

        self._tau = int(np.round(np.mean([x_mean, y_mean, z_mean])))

    @property
    def embedding_dimension(self) -> int:
        if self._dE is not None:
            return self._dE
        self._calculate_embedding_dimension()
        if self._dE is None:
            raise ValueError('Embedding dimension could not be calculated.')
        return self._dE

    def set_embedding_dimension(self, dE: int):
        self._dE = dE

    def _calculate_embedding_dimension(self):
        dE_max = self._embedding_dimensions.stack().max().astype(int)
        self._dE = dE_max

    def make_signals(self):
        self._signals = self._signals_from_participants()

    def compute_time_delays(self):
        if self._signals is None:
            self.make_signals()
        self._compute_time_delays()

    def compute_embedding_dimensions(self):
        if self._tau is None:
            self.compute_time_delays()
        self._compute_embedding_dimensions()

    def compute_state_spaces(self):
        if self._tau is None:
            self.compute_time_delays()
        if self._dE is None:
            self.compute_embedding_dimensions()
        self._compute_state_spaces()

    def compute_divergence_curves(self, force_recalculate: bool = False):
        if not force_recalculate and self._divergence_curves is not None:
            print('Divergence curves already computed. Loading from file. Use force_recalculate=True to recompute.')
            return
        if self._state_spaces is None:
            self.compute_state_spaces()
        self._compute_divergence_curves()

    def _load_signals(self) -> pd.DataFrame:
        try:
            return load_dataframe(self._path_signals)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Cannot load signals from file. {e}')

    def _signals_from_participants(self) -> pd.DataFrame:
        print('Computing signals from participants')
        if not self._participants:
            self._init_from_environment()
        df_signals = pd.DataFrame(columns=self.conditions, index=self.participant_ids, dtype=object)
        for participant in self._participants:
            for trial in participant.trials:
                df_signals.loc[participant.participant_id, trial.trial_id] = trial.data
        save_dataframe(df=df_signals, path_filename=self._path_signals)
        return df_signals

    def _compute_time_delays(self):
        print('Computing time delays')
        df_time_delays = process_on_dataframe(df=self._signals,
                                              func=minimum_average_mutual_information,
                                              multiprocess=not DEBUG,
                                              axes=['x', 'y', 'z'])
        save_dataframe(df=df_time_delays, path_filename=self._path_time_delays)
        self._time_delays = df_time_delays
        self._calculate_time_delay()

    @staticmethod
    def calc_embedding_dim(signal, tau):
        FNN_ratio, n = false_nearest_neighbours(signal=signal, delay=tau, stop_early=False)
        return 3 * (np.argmax(FNN_ratio - 0.01 < 0) + 1)

    def _compute_embedding_dimensions(self):
        print('Computing embedding dimensions')
        df_emb_dim = process_on_dataframe(df=self._signals,
                                          func=self.calc_embedding_dim,
                                          multiprocess=not DEBUG,
                                          tau=self._tau)
        save_dataframe(df=df_emb_dim, path_filename=self._path_embedding_dimensions)
        self._embedding_dimensions = df_emb_dim
        self._calculate_embedding_dimension()

    def _compute_state_spaces(self):
        print('Computing state spaces')
        df_state_spaces = process_on_dataframe(df=self._signals,
                                               func=state_space_reconstruction,
                                               multiprocess=not DEBUG,
                                               tau=self._tau,
                                               emb_dimension=self._dE,
                                               base_dim=3)
        save_dataframe(df=df_state_spaces, path_filename=self._path_state_spaces)
        self._state_spaces = df_state_spaces

    def _compute_divergence_curves(self):
        print('Computing divergence curves')
        df_div_curves = process_on_dataframe(df=self._state_spaces,
                                             func=rosenstein_divergence,
                                             multiprocess=not DEBUG)
        save_dataframe(df=df_div_curves, path_filename=self._path_divergence_curves)
        self._divergence_curves = df_div_curves

    def time_delay_summary(self):
        x_values = self.time_delays.apply(lambda col: col.map(lambda d: d.get('x') if isinstance(d, dict) else np.nan))
        y_values = self.time_delays.apply(lambda col: col.map(lambda d: d.get('y') if isinstance(d, dict) else np.nan))
        z_values = self.time_delays.apply(lambda col: col.map(lambda d: d.get('z') if isinstance(d, dict) else np.nan))
        x_mean = x_values.stack().mean()
        x_std = x_values.stack().std()
        y_mean = y_values.stack().mean()
        y_std = y_values.stack().std()
        z_mean = z_values.stack().mean()
        z_std = z_values.stack().std()
        # add plus minus sign
        print(f'Time delays: \nx={x_mean:.2f} +/- {x_std:.2f} \ny={y_mean:.2f} +/- {y_std:.2f} \nz={z_mean:.2f} +/- {z_std:.2f}')
        print(f'Time delay: {self._tau}')

    def embedding_dimension_summary(self):
        df_embedding_dim = self.embedding_dimensions
        dE_max = df_embedding_dim.stack().max().astype(int)
        dE_mean = df_embedding_dim.stack().mean()
        dE_std = df_embedding_dim.stack().std()

        print(f'{dE_max=}')
        print(f'{dE_mean=:.1f}')
        print(f'{dE_std=:.1f}')
        print(f'Embedding dimension: {self._dE}')

    def set_fit_interval(self, end: int, start: int = 0):
        self._fit_interval = (start, end)

    def compute_divergence_exponents(self):
        if self._divergence_curves is None:
            raise ValueError('Divergence curves must be computed first.')
        if self._fit_interval is None:
            raise ValueError('Fit interval must be set first.')
        self._compute_divergence_exponents()

    def _compute_divergence_exponents(self):
        print('Computing divergence exponents')
        df_divergence_exponents = process_on_dataframe(df=self._divergence_curves,
                                                       func=calculate_divergence_exponent,
                                                       multiprocess=not DEBUG,
                                                       fit_interval=self._fit_interval)
        save_dataframe(df=df_divergence_exponents, path_filename=self._path_divergence_exponents)
        self._divergence_exponents = df_divergence_exponents
