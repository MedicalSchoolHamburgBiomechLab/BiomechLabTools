import numpy as np


def state_space_reconstruction(signal: np.ndarray, tau: int, emb_dimension: int, base_dim: int):
    arr = np.zeros((len(signal), emb_dimension))

    if signal.ndim == 1:
        dim_in = 1
        signal = np.reshape(signal, (len(signal), 1))
    else:
        dim_in = signal.shape[1]

    for d in range(0, int(emb_dimension / dim_in)):
        arr[:, d * base_dim:(d * base_dim + base_dim)] = np.roll(signal, -1 * d * tau * base_dim)
    len_sig = len(signal) - tau * int((emb_dimension - base_dim) / base_dim)
    return arr[:len_sig, :]
