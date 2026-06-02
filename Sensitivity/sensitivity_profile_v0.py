"""
Plot the one dimensional sensitivity profile

Author: Martin Rädler
"""
# Python libraries
import sys
from matplotlib import pyplot as plt

# Auxiliary functions
from load_root_hist import load_tb_j_pet_brain_insert_th3d


def main():
    # Output directory from the post-processing
    output_dir = '/home/martin/J-PET/ROOT/Output'

    tbtb_true, tbbi_true, bibi_true, z_edges = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source/2025-12-31_15-27-48', '1_1_1_true', return_grid=True)

    tbtb_energy, tbbi_energy, bibi_energy = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source/2025-12-31_15-27-48', '1_1_1_energy')
    tbtb_energy_true, tbbi_energy_true, bibi_energy_true = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source/2025-12-31_15-27-48', '1_1_1_true_energy')

    tbtb_time, tbbi_time, bibi_time = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source/2025-12-31_15-27-48', '1_1_1_time')
    tbtb_time_true, tbbi_time_true, bibi_time_true = load_tb_j_pet_brain_insert_th3d(
        output_dir + '/Sensitivity_line_source/2025-12-31_15-27-48', '1_1_1_true_time')

    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    fig, ax = plt.subplots()
    # ax.stairs((tbtb_true + tbbi_true + bibi_true) / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:blue')
    # ax.stairs(tbtb_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:orange')
    # ax.stairs(tbbi_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:green')
    # ax.stairs(bibi_true / max_counts_per_bin, edges=z_edges, linestyle=':', color='tab:red')
    #
    # ax.stairs((tbtb_energy + tbbi_energy + bibi_energy) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    # ax.stairs(tbtb_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    # ax.stairs(tbbi_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    # ax.stairs(bibi_energy / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')
    #
    # ax.stairs((tbtb_energy_true + tbbi_energy_true + bibi_energy_true) / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    # ax.stairs(tbtb_energy_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    # ax.stairs(tbbi_energy_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    # ax.stairs(bibi_energy_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    ax.stairs((tbtb_time + tbbi_time + bibi_time) / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_time / max_counts_per_bin, edges=z_edges, linestyle='-', color='tab:red')

    ax.stairs((tbtb_time_true + tbbi_time_true + bibi_time_true) / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_time_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_time_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_time_true / max_counts_per_bin, edges=z_edges, linestyle='--', color='tab:red')

    ax.set_ylim(0, 55)

    plt.show()

    fig, ax = plt.subplots()

    ax.stairs((tbtb_time_true + tbbi_time_true + bibi_time_true) / (tbtb_time + tbbi_time + bibi_time), edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_time_true / tbtb_time, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_time_true / tbbi_time, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_time_true / bibi_time, edges=z_edges, linestyle='--', color='tab:red')

    ax.stairs((tbtb_energy_true + tbbi_energy_true + bibi_energy_true) / (tbtb_energy + tbbi_energy + bibi_energy), edges=z_edges, linestyle='-', color='tab:blue')
    ax.stairs(tbtb_energy_true / tbtb_energy, edges=z_edges, linestyle='-', color='tab:orange')
    ax.stairs(tbbi_energy_true / tbbi_energy, edges=z_edges, linestyle='-', color='tab:green')
    ax.stairs(bibi_energy_true / bibi_energy, edges=z_edges, linestyle='-', color='tab:red')

    plt.show()

    return 0


if __name__ == "__main__":
    main()
