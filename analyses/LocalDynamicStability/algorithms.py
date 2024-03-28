import warnings

import numpy as np


def rosenstein_divergence(x: np.ndarray,
                          period: int = 100,
                          truncate: int = 0,
                          return_initial_distance_only: bool = False):
    """
    Calculate the average logarithm of the distance between a point and its nearest neighbour in a reconstructed state space.
    :param x: reconstructed state space
    :param period: in cyclic motions: average number of samples of one cycle (e.g. 100 strides in 10,000 samples -> period = 100)
    :param truncate: when multiple bouts of cyclic motion are concatenated (e.g. shuttle runs/walking bouts), this defines the number of samples in one bout.
    :param return_initial_distance_only: convenience flag to return only the initial distance.
    :return: The average logarithmic rate of divergence in the state space.
    """
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Rosenstein at al. 1995: "A practical method for calculating largest Lyapunov exponents from small data sets".
    # 10.1016/0167-2789(93)90009-P
    # # # # # # # # # # # # # # # # # # # # # # # #
    futures = []
    data = []
    for i in range(len(x)):
        data.append(get_distance_list(x, i, period, True, truncate))
    # preallocate array for ln values
    distance_array = np.empty((len(x), len(x) - period))
    distance_array[:] = np.nan
    for i, dd in enumerate(data):
        distance_array[i, :len(dd)] = np.array(dd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress "Mean of empty slice" RunTimeWarning
        distance_avg = np.nanmean(distance_array, axis=0)

    ln_avg = np.log(distance_avg)
    ln_avg = ln_avg[~np.isnan(ln_avg)]  # drop nan values
    # return period + 1 length for a nicer display...
    if return_initial_distance_only:
        return np.nanmean(distance_array[:, 0])
    return ln_avg[:period + 1]


def get_distance_list(x, i, period, early_stop: bool = True, truncate: int = 0):
    distance = []
    # find initial nearest neighbour
    x_0 = x[i]
    # Stipulation: Neighbouring point must have a temporal distance greater than the mean period.
    # Resulting in "pot_inn" potential initial nearest neighbours
    pot_inn = x.copy()
    start_exclude = np.max((0, (i - period - 1)))
    stop_exclude = np.min((len(x), i + period))
    pot_inn[start_exclude:stop_exclude] = np.nan  # setting those which cannot be initial NN to nan, so they're not chosen

    delta = np.linalg.norm(pot_inn - x_0, axis=1)
    i_nn = np.nanargmin(delta)

    distance.append(delta[i_nn])
    k = 1
    # define maximum samples to follow.
    k_max = len(x) - np.max((i, i_nn)) - 1
    if truncate > 0:
        # wordy just for clarity's sake, in case of "shuttle runs", where separate shuttles are concatenated
        # we must not follow trajectories across the concatenation boundaries.
        # maximum samples to follow from current value until next truncation/concatenation:
        k_max_x_0 = truncate - (i % truncate)
        # maximum samples to follow from initial nearest neighbour until next truncation/concatenation:
        k_max_inn = truncate - (i_nn % truncate)
        k_max = np.min((k_max, k_max_x_0, k_max_inn)) - 1
    if early_stop:
        # stop after "period"
        k_max = np.min((k_max, period + 1))
    if k_max < 3:  # different suggestions welcome...
        # skip if too few values are expected
        return []
    while k <= k_max:
        dist = np.linalg.norm(x[i + k] - x[i_nn + k])
        distance.append(dist)
        k += 1
    distance = np.array(distance)
    return distance
