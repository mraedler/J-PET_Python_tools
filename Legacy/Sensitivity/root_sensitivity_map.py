"""
Analyze the sensitivity map calculated in root

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from uproot import open as open_root
from matplotlib.colors import ListedColormap, BoundaryNorm

# Auxiliary functions
from vis import vis_3d


def main():
    output_directory = '/home/martin/J-PET/ROOT/Output'
    simulations = ['/2025-10-02_15-28-11',
                   '/2025-10-03_15-17-17',
                   '/2025-10-07_15-28-09',
                   '/2025-10-10_17-44-45',
                   '/2025-10-21_11-20-21',
                   '/2025-10-23_15-26-48',
                   '/2025-12-05_15-06-14']

    # root_file = open_root(output_directory + simulations[0] + '/merged_1_1_1.root')
    root_file = open_root(output_directory + simulations[0] + '/merged_8_8_10.root')
    _, x_edges, y_edges, z_edges = root_file['TH3'].to_numpy()
    arr_acc = np.zeros((x_edges.size - 1, y_edges.size - 1, z_edges.size - 1))

    for ii in range(0, len(simulations)):
        # root_file = open_root(output_directory + simulations[ii] + '/merged_1_1_1.root')
        root_file = open_root(output_directory + simulations[ii] + '/merged_8_8_10.root')
        # print(root_file.keys())
        arr, _, _, _ = root_file['TH3'].to_numpy()
        arr_acc += arr
        root_file.close()

    arr_acc_slice = arr_acc[:, :, int(arr_acc.shape[2] / 2)]
    # arr_acc_slice = arr_acc[:, :, -1]

    plot_radial_profiles(x_edges, y_edges, z_edges, arr_acc)





    # vis_3d(arr_acc, spacing=[1, 1, 1])
    fig, ax = plt.subplots()
    # im = ax.imshow(arr_acc_slice.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    im = ax.imshow(arr_acc_slice.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], clim=(0, 1e5))
    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')
    c_bar = plt.colorbar(im)
    c_bar.set_label('Counts')
    c_bar.formatter.set_powerlimits((0, 0))
    plt.show()


    z_profile = np.sum(arr_acc, axis=(0, 1))

    fig, ax = plt.subplots()
    ax.stairs(z_profile, edges=z_edges)
    plt.show()

    return 0


def plot_radial_profiles(x_edges, y_edges, z_edges, arr_acc):
    x = (x_edges[1:] + x_edges[:-1]) / 2
    y = (y_edges[1:] + y_edges[:-1]) / 2
    z = (z_edges[1:] + z_edges[:-1]) / 2

    n_phi = 48
    phi = np.arange(n_phi) / n_phi * 2 * np.pi
    rho = np.linspace(0, 120, 121)
    rho_mesh, phi_mesh = np.meshgrid(rho, phi, indexing='ij')
    x_itp, y_itp = rho_mesh * np.cos(phi_mesh), rho_mesh * np.sin(phi_mesh)

    n = arr_acc.shape[-1]
    # profiles = np.zeros((rho.size, n))
    cmap = plt.get_cmap("viridis", n + 1)  # discrete version of viridis
    norm = BoundaryNorm(z_edges, cmap.N)

    fig, ax = plt.subplots()
    # for ii in range(n):
    for ii in np.arange(n):
        interpolator = RegularGridInterpolator((x, y), arr_acc[:, :, ii])
        arr_acc_itp = interpolator((x_itp, y_itp))
        # profiles[:, ii] = np.mean(arr_acc_itp, axis=1)

        ax.plot(rho, np.mean(arr_acc_itp, axis=1), color=cmap(norm(z[ii])))
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$z$ [mm]')
    ax.set_xlabel('Radius [mm]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylabel('Counts')
    plt.show()
    return 0


if __name__ == "__main__":
    main()
