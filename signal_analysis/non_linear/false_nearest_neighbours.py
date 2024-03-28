import warnings

import numpy as np
from scipy.spatial import KDTree

from .delay_coordinate_embedding import state_space_reconstruction


def false_nearest_neighbours(signal: np.ndarray,
                             delay: int,
                             max_dim: int = 18,
                             r_tol: float = 10.0,
                             a_tol: float = 2.0,
                             threshold: float = 0.1,
                             stop_early: bool = True):
    # Kennel et al. 1992: "Determining embedding dimension for phase-space reconstruction using a
    # geometrical construction"
    # Adopted from: https://github.com/TeaspoonTDA/teaspoon/blob/master/teaspoon/teaspoon/parameter_selection/FNN_n.py
    s_i_std = np.std(signal)
    Xfnn = []
    dim_array = []
    min_dim = 0

    if signal.ndim == 1:
        dim_in = 1
    else:
        dim_in = signal.shape[1]

    for dim in range(dim_in, max_dim, dim_in):
        s_i = state_space_reconstruction(signal, tau=delay, emb_dimension=dim, base_dim=dim_in)
        tree = KDTree(s_i)
        R, i = tree.query(s_i, k=2)

        if dim > dim_in:
            R_dp1 = np.sqrt(np.sum((np.square(s_i[ind, :] - s_i[i_nn, :])), axis=1))
            # Criterion 1 : increase in distance between neighbors is large
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress runtime warnings
                num1 = np.heaviside(np.divide(abs(s_i[i_nn, -1] - s_i[ind, -1]), dp1) - r_tol, 0.5)

            # Criterion 2 : nearest neighbor not necessarily close to y(n)
            num2 = np.heaviside(a_tol - R_dp1 / s_i_std, 0.5)
            num = sum(np.multiply(num1, num2))
            den = sum(num2)
            ratio = (num / den)
            Xfnn.append(ratio)
            dim_array.append(dim - 1)
            # print(f'Dimension: {dim}, Percent false nearest neighbours: {ratio*100}')

            # break criterion
            if ratio <= threshold:
                min_dim = dim - 1
                if stop_early:
                    break

        # # # save the result of this dimension for the next iteration (signal length will be shorter then)
        len_dp1 = len(signal) - delay * int(dim / dim_in)
        # neglect the last samples (they won't exist in the next iteration)
        dp1 = R[:len_dp1, 1]  # distance
        i_nn = i[:len_dp1, 1]  # indices of the nearest neighbours in this iteration's dimension
        ind = i_nn <= len_dp1 - 1  # filter those samples whose nearest neighbours won't exist in the next iteration
        i_nn = i_nn[ind]  # also filter the nearest neighbours themselves
        dp1 = dp1[ind]
    Xfnn = np.array(Xfnn)
    return Xfnn, min_dim
