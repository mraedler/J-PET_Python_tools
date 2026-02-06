"""
Accumulate the sensitivity maps generated in ROOT

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from glob import glob
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from uproot import open as open_root
from matplotlib.colors import BoundaryNorm

# Auxiliary functions
from vis import vis_3d
from write_interfile import write_sensitivity_map


def main():
    # gantry = 'Comb.'
    # gantry = 'TB-TB'
    # gantry = 'TB-BI'
    gantry = 'BI-BI'

    root_file_paths = glob('/home/martin/J-PET/ROOT/Output/Sensitivity_brain_insert_4_18/*/' + gantry + '_8_8_10.root')
    n_runs = len(root_file_paths)  # Simulating each 10 000 seconds of a 1 MBq source

    root_file = open_root(root_file_paths[0])
    _, x_edges, y_edges, z_edges = root_file['TH3'].to_numpy()
    arr_acc = np.zeros((x_edges.size - 1, y_edges.size - 1, z_edges.size - 1))

    maximum_voxel_count = estimate_maximum_voxel_count(x_edges, y_edges, z_edges, n_runs)

    for ii in range(0, len(root_file_paths)):
        root_file = open_root(root_file_paths[ii])
        # print(root_file.keys())
        arr, _, _, _ = root_file['TH3'].to_numpy()
        arr_acc += arr
        root_file.close()

    arr_acc /= maximum_voxel_count

    vis_3d(arr_acc)

    inside_circle = intersects_with_circle(x_edges, y_edges, (0., 0.), 144.)
    red_img = np.ones((x_edges.size - 1, y_edges.size - 1, 3)) * np.array([1, 0, 0], ndmin=3)
    inside_circle_img = np.dstack((red_img, inside_circle.astype(float) * 0.5))

    mask = np.repeat(inside_circle[:, :, np.newaxis].astype(float), z_edges.size, axis=2)

    vis_3d(mask)

    # write_sensitivity_map(x_edges, y_edges, z_edges, arr_acc,
    #                       '/home/martin/J-PET/CASToR_RECONS/SENS_MAPS/Brain_insert_4_18/',
    #                       gantry, 'PET_7th_gen_brain_insert_dz_1_mm')

    write_sensitivity_map(x_edges, y_edges, z_edges, mask,
                          '/home/martin/J-PET/CASToR_RECONS/MASKS/',
                          'Brain_insert', 'PET_7th_gen_brain_insert_dz_1_mm')

    sys.exit()

    arr_acc_slice = arr_acc[:, :, int(arr_acc.shape[2] / 2)]
    # arr_acc_slice = arr_acc[:, :, -1]

    # plot_radial_profiles(x_edges, y_edges, z_edges, arr_acc)

    # vis_3d(arr_acc, spacing=[1, 1, 1])
    fig, ax = plt.subplots()
    im = ax.imshow(arr_acc_slice.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # im = ax.imshow(arr_acc_slice.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], clim=(0, 1e5))
    ax.imshow(inside_circle_img, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')
    ax.add_patch(plt.Circle((0, 0), 144, facecolor='none', edgecolor='r'))
    c_bar = plt.colorbar(im)
    c_bar.set_label('Counts')
    c_bar.formatter.set_powerlimits((0, 0))
    plt.show()


    z_profile = np.sum(arr_acc, axis=(0, 1))

    fig, ax = plt.subplots()
    ax.stairs(z_profile, edges=z_edges)
    plt.show()

    return 0


def estimate_maximum_voxel_count(x_edges, y_edges, z_edges, n_runs):
    dx, dy, dz = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0], z_edges[1] - z_edges[0]  # [mm], [mm], [mm]

    cylinder_radius = (x_edges[-1] - x_edges[0]) / 2  # [mm]
    cylinder_height = z_edges[-1] - z_edges[0]  # [mm]
    cylinder_volume = np.pi * cylinder_radius ** 2 * cylinder_height  # [mm^3]

    # Assuming 10 000 seconds per run with an activity of 1 MBq
    activity_concentration = n_runs * 1e4 * 1e6 / cylinder_volume  # [1/mm^3]

    maximum_voxel_count = activity_concentration * dx * dy * dz

    return maximum_voxel_count


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


def intersects_with_circle(x_edges, y_edges, circle_center, circle_radius, visualize=False):
    #
    x_centers, y_centers = (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2
    x_centers_mesh, y_centers_mesh = np.meshgrid(x_centers, y_centers, indexing='ij')

    # # Corner closest to the circle center
    # p_x = np.minimum(np.maximum(circle_center[0], x_edges[:-1]), x_edges[1:])
    # p_y = np.minimum(np.maximum(circle_center[1], y_edges[:-1]), y_edges[1:])

    # Corner furthest away from the circle center
    p_x = np.where(np.abs(circle_center[0] - x_edges[:-1]) > np.abs(circle_center[0] - x_edges[1:]), x_edges[:-1], x_edges[1:])
    p_y = np.where(np.abs(circle_center[1] - y_edges[:-1]) > np.abs(circle_center[1] - y_edges[1:]), y_edges[:-1], y_edges[1:])

    #
    p_x_mesh, p_y_mesh = np.meshgrid(p_x, p_y, indexing='ij')
    d_squared = (circle_center[0] - p_x_mesh) ** 2 + (circle_center[1] - p_y_mesh) ** 2
    inside_circle = d_squared <= circle_radius ** 2

    if visualize:
        # Flatten to vectors for the plot
        x_centers_mesh, y_centers_mesh = x_centers_mesh.flatten(), y_centers_mesh.flatten()
        p_x_mesh, p_y_mesh = p_x_mesh.flatten(), p_y_mesh.flatten()

        fig, ax = plt.subplots(1, 1)
        ax.imshow(inside_circle.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        ax.plot(np.vstack((x_centers_mesh, p_x_mesh)), np.vstack((y_centers_mesh, p_y_mesh)), '-o')
        ax.add_patch(plt.Circle((0, 0), 144, facecolor='none', edgecolor='r'))
        ax.set_xticks(x_edges)
        ax.set_yticks(y_edges)
        ax.grid(True)
        plt.show()

    return inside_circle


if __name__ == "__main__":
    main()
