import numpy as np


def lorenz(x, y, z, sigma, rho, beta):
    """
    Given:
       x, y, z: a point of interest in three-dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot


def gen_time_series(n_steps: int = 10000, dt: float = 0.01,
                    x_0: float = 5.57, y_0: float = -6.74, z_0: float = 51.85,
                    sigma: float = 16.0, rho: float = 45.92, beta: float = 4.0):
    # preallocate arrays
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    xs = np.empty(n_steps + 1)
    ys = np.empty(n_steps + 1)
    zs = np.empty(n_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = (x_0, y_0, z_0)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(n_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], sigma=sigma, rho=rho, beta=beta)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    return np.vstack((t, xs, ys, zs))


LorenzSystem = gen_time_series()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))

    # prepare the axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plot the Lorenz attractor
    ax.plot(LorenzSystem[1], LorenzSystem[2], LorenzSystem[3], 'b', alpha=0.7)

    plt.show()
