"""
Uncertainty estimation with respect to the point of interaction (POI)

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions


def main():
    # Crystal dimensions
    delta_x = 30  # [mm]
    delta_y = 6  # [mm]
    delta_z = 3  # [mm]

    # Transformation parameters
    x1, y1, _ = get_pois(100000, delta_x, delta_y, delta_z, xx=200.)
    x2, y2, _ = get_pois(100000, delta_x, delta_y, delta_z, xx=-200.)

    # fig, ax = plt.subplots()
    # # ax.scatter(x1, y1)
    # # ax.scatter(x2, y2)
    # ax.plot(np.stack((x1, x2)), np.stack((y1, y2)), color='tab:blue', alpha=0.5)
    # # ax.set_aspect(1)
    # plt.show()

    x_edges, x_centers, x_widths = get_bins(100, delta=.15)
    y_edges, y_centers, y_widths = get_bins(100, delta=.15)

    h_x, _ = np.histogram((x1 + x2) / 2, bins=x_edges)
    # h_y, _ = np.histogram((y1 + y2) / 2, bins=y_edges)
    h_y, _ = np.histogram(y1 * 0.5 + y2 * 0.5, bins=y_edges)

    fig, ax = plt.subplots()
    # ax.bar(x_centers, h_x, width=x_widths)
    ax.bar(y_centers, h_y, width=y_widths)
    plt.show()

    return 0


def get_pois(n_samples, delta_x, delta_y, delta_z, theta=0., xx=0., yy=0., zz=0.):
    # Sample uniformly
    x = np.random.uniform(low=-delta_x/2, high=delta_x/2, size=n_samples)
    y = np.random.uniform(low=-delta_y/2, high=delta_y/2, size=n_samples)
    z = np.random.uniform(low=-delta_z/2, high=delta_z/2, size=n_samples)

    # Apply transforms
    xp = x * np.cos(theta) - y * np.sin(theta) + xx
    yp = x * np.sin(theta) + y * np.cos(theta) + yy
    zp = z + zz

    # samples = np.stack((x, y, z), axis=1)
    return xp, yp, zp


def get_bins(n, delta=1.):
    bin_edges = (np.arange(n + 1) - n / 2) * delta
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    return bin_edges, bin_centers, bin_widths


if __name__ == "__main__":
    main()
