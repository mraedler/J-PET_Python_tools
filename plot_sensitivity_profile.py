"""
Plot the one dimensional sensitivity profile

Author: Martin Rädler
"""
# Python libraries
import sys
from glob import glob
from uproot import open as open_root
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Load the sensitivity profiles
    x_edges, y_edges, z_edges, comb_true = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/Comb._1_1_1_true.root'))
    _, _, _, tbtb_true = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-TB_1_1_1_true.root'))
    _, _, _, tbbi_true = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-BI_1_1_1_true.root'))
    _, _, _, bibi_true = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/BI-BI_1_1_1_true.root'))
    comb_true, tbtb_true, tbbi_true, bibi_true = comb_true.flatten(), tbtb_true.flatten(), tbbi_true.flatten(), bibi_true.flatten()

    _, _, _, comb_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/Comb._1_1_1_time.root'))
    _, _, _, tbtb_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-TB_1_1_1_time.root'))
    _, _, _, tbbi_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-BI_1_1_1_time.root'))
    _, _, _, bibi_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/BI-BI_1_1_1_time.root'))
    comb_time, tbtb_time, tbbi_time, bibi_time = comb_time.flatten(), tbtb_time.flatten(), tbbi_time.flatten(), bibi_time.flatten()

    _, _, _, comb_true_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/Comb._1_1_1_true_time.root'))
    _, _, _, tbtb_true_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-TB_1_1_1_true_time.root'))
    _, _, _, tbbi_true_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-BI_1_1_1_true_time.root'))
    _, _, _, bibi_true_time = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/BI-BI_1_1_1_true_time.root'))
    comb_true_time, tbtb_true_time, tbbi_true_time, bibi_true_time = comb_true_time.flatten(), tbtb_true_time.flatten(), tbbi_true_time.flatten(), bibi_true_time.flatten()

    _, _, _, comb_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/Comb._1_1_1_energy.root'))
    _, _, _, tbtb_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-TB_1_1_1_energy.root'))
    _, _, _, tbbi_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-BI_1_1_1_energy.root'))
    _, _, _, bibi_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/BI-BI_1_1_1_energy.root'))
    comb_energy, tbtb_energy, tbbi_energy, bibi_energy = comb_energy.flatten(), tbtb_energy.flatten(), tbbi_energy.flatten(), bibi_energy.flatten()

    _, _, _, comb_true_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/Comb._1_1_1_true_energy.root'))
    _, _, _, tbtb_true_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-TB_1_1_1_true_energy.root'))
    _, _, _, tbbi_true_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/TB-BI_1_1_1_true_energy.root'))
    _, _, _, bibi_true_energy = load_root_th3(glob('/home/martin/J-PET/ROOT/Output/Sensitivity_line_source/*/BI-BI_1_1_1_true_energy.root'))
    comb_true_energy, tbtb_true_energy, tbbi_true_energy, bibi_true_energy = comb_true_energy.flatten(), tbtb_true_energy.flatten(), tbbi_true_energy.flatten(), bibi_true_energy.flatten()

    # Subsample
    f_sub = 4  # 2, 2, 5, 127 or combinations of those
    z_edges = z_edges[::f_sub]
    comb_true, tbtb_true, tbbi_true, bibi_true = subsample(comb_true, f_sub), subsample(tbtb_true, f_sub), subsample(tbbi_true, f_sub), subsample(bibi_true, f_sub)
    comb_time, tbtb_time, tbbi_time, bibi_time = subsample(comb_time, f_sub), subsample(tbtb_time, f_sub), subsample(tbbi_time, f_sub), subsample(bibi_time, f_sub)
    comb_true_time, tbtb_true_time, tbbi_true_time, bibi_true_time = subsample(comb_true_time, f_sub), subsample(tbtb_true_time, f_sub), subsample(tbbi_true_time, f_sub), subsample(bibi_true_time, f_sub)
    comb_energy, tbtb_energy, tbbi_energy, bibi_energy = subsample(comb_energy, f_sub), subsample(tbtb_energy, f_sub), subsample(tbbi_energy, f_sub), subsample(bibi_energy, f_sub)
    comb_true_energy, tbtb_true_energy, tbbi_true_energy, bibi_true_energy = subsample(comb_true_energy, f_sub), subsample(tbtb_true_energy, f_sub), subsample(tbbi_true_energy, f_sub), subsample(bibi_true_energy, f_sub)

    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    fig, ax = plt.subplots()
    # ax.stairs(comb_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:blue')
    # ax.stairs(tbtb_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:orange')
    # ax.stairs(tbbi_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:green')
    # ax.stairs(bibi_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:red')

    # ax.stairs(comb_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    # ax.stairs(tbtb_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    # ax.stairs(tbbi_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    # ax.stairs(bibi_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    # ax.stairs(comb_true_energy / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    # ax.stairs(tbtb_true_energy / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    # ax.stairs(tbbi_true_energy / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    # ax.stairs(bibi_true_energy / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    ax.stairs(comb_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs(comb_true_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_true_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_true_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_true_time / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    ax.set_ylim(0, 55)

    plt.show()

    fig, ax = plt.subplots()

    ax.stairs(comb_true_time / comb_time, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_true_time / tbtb_time, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_true_time / tbbi_time, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_true_time / bibi_time, edges=z_edges, linestyle='--', color='tab:red')

    ax.stairs(comb_true_energy / comb_energy, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_true_energy / tbtb_energy, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_true_energy / tbbi_energy, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_true_energy / bibi_energy, edges=z_edges, linestyle='-', color='tab:red')

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
