"""

"""
import sys

import numpy as np
from uproot import open as open_root


def load_root_th3(root_file_paths_list):
    # Get the coordinate grid
    root_file = open_root(root_file_paths_list[0])
    _, x_edges, y_edges, z_edges = root_file['TH3'].to_numpy()
    th3_accumulated = np.zeros((x_edges.size - 1, y_edges.size - 1, z_edges.size - 1))

    # Accumulate the data
    for ii in range(0, len(root_file_paths_list)):
        root_file = open_root(root_file_paths_list[ii])
        # print(root_file.keys())
        arr, _, _, _ = root_file['TH3'].to_numpy()
        th3_accumulated += arr
        root_file.close()

    return x_edges, y_edges, z_edges, th3_accumulated


def load_th3d(root_file_path, return_grid=False):
    # Get the coordinate grid
    root_file = open_root(root_file_path)
    arr, x_edges, y_edges, z_edges = root_file['TH3'].to_numpy()

    if return_grid:
        return arr, x_edges, y_edges, z_edges
    else:
        return arr


def load_tb_j_pet_brain_insert_th3d(root_files_directory, selection, z_sub=4, return_grid=False):
    # Load
    tbtb, _, _, z_bin_edges = load_th3d(root_files_directory + '/TBTB_' + selection + '.root', return_grid=True)
    tbbi = load_th3d(root_files_directory + '/TBBI_' + selection + '.root', return_grid=False)
    bibi = load_th3d(root_files_directory + '/BIBI_' + selection + '.root', return_grid=False)

    # Reduce to 1D
    tbtb, tbbi, bibi = tbtb.flatten(), tbbi.flatten(), bibi.flatten()

    # Sub-sample
    z_bin_edges = z_bin_edges[::z_sub]
    tbtb, tbbi, bibi = subsample(tbtb, z_sub), subsample(tbbi, z_sub), subsample(bibi, z_sub)

    if return_grid:
        return tbtb, tbbi, bibi, z_bin_edges
    else:
        return tbtb, tbbi, bibi

    # sys.exit()
    #
    #
    #
    # # Get the coordinate grid
    # root_file = open_root(root_files_directory + '/TBTB_' + selection + '.root')
    # tbtb, x_edges, y_edges, z_edges = root_file['TH3'].to_numpy()
    #
    # root_file = open_root(root_files_directory + 'TBBI_' + selection + '.root')
    # tbbi, _, _, _ = root_file['TH3'].to_numpy()
    #
    # root_file = open_root(root_files_directory + 'BIBI_' + selection + '.root')
    # bibi, _, _, _ = root_file['TH3'].to_numpy()
    #
    # z_edges = z_edges[::z_sub]
    # tbtb, tbbi, bibi = subsample(tbtb, z_sub), subsample(tbbi, z_sub), subsample(bibi, z_sub)
    #
    # if return_grid:
    #     return x_edges, y_edges, z_edges, tbtb, tbbi, bibi
    # else:
    #     return tbtb, tbbi, bibi


def load_tb_j_pet_brain_insert_thnd(root_files_directory, selection, z_sub=4, return_grid=False):
    # Load
    bin_edges, _, tbtb = load_thnd(root_files_directory + '/TBTB_' + selection + '.root', return_grid=True)
    tbbi = load_thnd(root_files_directory + '/TBBI_' + selection + '.root', return_grid=False)
    bibi = load_thnd(root_files_directory + '/BIBI_' + selection + '.root', return_grid=False)

    # Reduce to 1D
    z_bin_edges = bin_edges[-1]
    tbtb, tbbi, bibi = tbtb.flatten(), tbbi.flatten(), bibi.flatten()

    # Sub-sample
    z_bin_edges = z_bin_edges[::z_sub]
    tbtb, tbbi, bibi = subsample(tbtb, z_sub), subsample(tbbi, z_sub), subsample(bibi, z_sub)

    if return_grid:
        return tbtb, tbbi, bibi, z_bin_edges
    else:
        return tbtb, tbbi, bibi


def subsample(a, m):
    # a.size must be divisible by m
    return a.reshape(-1, m).sum(axis=1)


def load_thnd(root_file_path, return_grid=False):
    root_file = open_root(root_file_path)
    data = root_file['THnD'].tojson()

    #
    # print(data.keys())
    n_dim = data['fNdimensions']
    n_entries = data['fEntries']

    # Axes
    n_bins = np.array([data['fAxes']['arr'][d]['fNbins'] for d in range(n_dim)])
    x_min = np.array([data['fAxes']['arr'][d]['fXmin'] for d in range(n_dim)])
    x_max = np.array([data['fAxes']['arr'][d]['fXmax'] for d in range(n_dim)])

    bin_edges = [np.linspace(x_min[ii], x_max[ii], n_bins[ii] + 1) for ii in range(n_bins.size)]
    bin_centers = [(bin_edges[ii][:-1] + bin_edges[ii][1:]) / 2 for ii in range(n_bins.size)]

    # Array
    arr = np.array(data['fArray']['fData']).reshape(n_bins + 2)

    # Alternatively load with the stride
    # print(data['fArray']['_typename'])
    # arr = np.array(data["fArray"]['fData'])
    # index_strides = np.array(data['fArray']['fSizes'])
    # byte_strides = tuple(index_strides[1:])
    # arr = np.lib.stride_tricks.as_strided(arr, shape=n_bins + 2, strides=byte_strides)

    # Remove the extra dimensions that collect entries outside the histogram
    # print((arr[0] + arr[-1]) / np.sum(arr[1:-1]))
    arr = arr[(slice(1, -1),) * n_dim]
    # arr = arr.squeeze()

    # Remove the list if arr is 1D
    if len(arr.shape) == 1:
        bin_edges = bin_edges[0]
        bin_centers = bin_centers[0]

    if return_grid:
        return bin_edges, bin_centers, arr
    else:
        return arr
