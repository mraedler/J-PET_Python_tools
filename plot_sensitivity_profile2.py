"""
Plot the one dimensional sensitivity profile

Author: Martin Rädler
"""
# Python libraries
import sys
from glob import glob
import numpy as np
from uproot import open as open_root
import matplotlib.pyplot as plt


def main():

    x_edges, y_edges, z_edges, tbtb_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBTB_true.root'))
    _, _, _, tbbi_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBBI_true.root'))
    _, _, _, bibi_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/BIBI_true.root'))

    _, _, _, tbtb_time = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBTB_time.root'))
    _, _, _, tbbi_time = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBBI_time.root'))
    _, _, _, bibi_time = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/BIBI_time.root'))

    _, _, _, tbtb_energy = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBTB_energy.root'))
    _, _, _, tbbi_energy = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBBI_energy.root'))
    _, _, _, bibi_energy = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/BIBI_energy.root'))

    _, _, _, tbtb_time_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBTB_time_true.root'))
    _, _, _, tbbi_time_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBBI_time_true.root'))
    _, _, _, bibi_time_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/BIBI_time_true.root'))

    _, _, _, tbtb_energy_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBTB_energy_true.root'))
    _, _, _, tbbi_energy_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/TBBI_energy_true.root'))
    _, _, _, bibi_energy_true = load_root_th3(glob('/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output/Sensitivity_line_source_200_ps/*/BIBI_energy_true.root'))

    # Subsample
    f_sub = 4  # 2, 2, 5, 127 or multiplicative combinations of those
    z_edges = z_edges[::f_sub]
    tbtb_true, tbbi_true, bibi_true = subsample(tbtb_true, f_sub), subsample(tbbi_true, f_sub), subsample(bibi_true, f_sub)
    tbtb_time, tbbi_time, bibi_time = subsample(tbtb_time, f_sub), subsample(tbbi_time, f_sub), subsample(bibi_time, f_sub)
    tbtb_energy, tbbi_energy, bibi_energy = subsample(tbtb_energy, f_sub), subsample(tbbi_energy, f_sub), subsample(bibi_energy, f_sub)
    tbtb_time_true, tbbi_time_true, bibi_time_true = subsample(tbtb_time_true, f_sub), subsample(tbbi_time_true, f_sub), subsample(bibi_time_true, f_sub)
    tbtb_energy_true, tbbi_energy_true, bibi_energy_true = subsample(tbtb_energy_true, f_sub), subsample(tbbi_energy_true, f_sub), subsample(bibi_energy_true, f_sub)

    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    fig, ax = plt.subplots()
    ax.stairs((tbtb_true + tbbi_true + bibi_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs((tbtb_time + tbbi_time + bibi_time) / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    ax.stairs((tbtb_energy + tbbi_energy + bibi_energy) / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:blue')
    ax.stairs(tbtb_energy / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:orange')
    ax.stairs(tbbi_energy / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:green')
    ax.stairs(bibi_energy / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:red')
    plt.show()

    fig, ax = plt.subplots()
    ax.stairs((tbtb_true + tbbi_true + bibi_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs((tbtb_energy_true + tbbi_energy_true + bibi_energy_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_energy_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_energy_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_energy_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs((tbtb_time_true + tbbi_time_true + bibi_time_true) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_time_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_time_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_time_true / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    plt.show()

    fig, ax = plt.subplots()
    ax.stairs((tbtb_time_true + tbbi_time_true + bibi_time_true) / (tbtb_time + tbbi_time + bibi_time), edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_time_true / tbtb_time, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_time_true / tbbi_time, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_time_true / bibi_time, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs((tbtb_energy_true + tbbi_energy_true + bibi_energy_true) / (tbtb_energy + tbbi_energy + bibi_energy), edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_energy_true / tbtb_energy, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_energy_true / tbbi_energy, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_energy_true / bibi_energy, edges=z_edges, linestyle='--', color='tab:red')
    plt.show()

    return 0


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


def subsample(a, m):
    # a.size must be divisible by m
    return a.reshape(-1, m).sum(axis=1)


if __name__ == "__main__":
    main()
