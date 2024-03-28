import numpy as np


def mutual_information(s: np.ndarray, q: np.ndarray):
    def histedges_equalN(x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                         np.arange(npt),
                         np.sort(x))

    # TODO: Implement algorithm to determine optimum number of bins
    #  Cellucci et al. (2005): "Statistical validation of mutual information calculations:
    #  Comparison of alternative numerical algorithms"
    # define the number of elements (i.e. 2D-bins) to fulfill the requirement that at
    # least 5 data-pairs are present in each element
    n_elements = int(np.sqrt(len(s) / 5))

    # plt.hist(s, histedges_equalN(s, n_elements))
    # plt.hist2d(s, q, [histedges_equalN(s, n_elements), histedges_equalN(q, n_elements)])
    # plt.axis('equal')
    # sns.jointplot(s, q).plot_joint(sns.histplot, alpha=0.7, color=[.9, .2, .5])
    # sns.set()
    # plt.show()

    n_bins = n_elements
    count_s, bins_s = np.histogram(s, n_bins)
    count_q, bins_q = np.histogram(q, n_bins)
    occupancy, _, _ = np.histogram2d(s, q, bins=[n_bins, n_bins])
    joint_prob = occupancy / np.sum(occupancy)
    I_xy = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            P_xy = joint_prob[i, j]
            P_x = count_s[i] / len(s)
            P_y = count_q[j] / len(q)
            if P_xy != 0:
                add = P_xy * np.log2((P_xy / (P_x * P_y)))
                I_xy += add
    # TODO: Implement bias correction from
    #  Roulston (1999): Estimating the errors on measured entropy and mutual information
    return I_xy


def minimum_average_mutual_information(s: np.ndarray, axes: list = None):
    def compute_mi(s):
        mi = []
        lag = 0
        while lag < len(s):
            # create time shifted copy of the original time series
            q = np.roll(s, lag)
            # calculate mutual information of original signal and its copy
            I_xy = mutual_information(s, q)
            mi.append(I_xy)
            lag += 1
            # return at the first minimum
            if np.diff(mi).size > 0:
                if np.diff(mi)[-1] > 0:
                    return lag - 1

    if axes is None:
        # If axes is not provided, execute once and return as a scalar
        return compute_mi(s)
    else:
        # If axes is provided, loop over the list elements and return a list
        result = {}
        for a, axis in enumerate(axes):
            result[axis] = compute_mi(s[:, a])
        return result
