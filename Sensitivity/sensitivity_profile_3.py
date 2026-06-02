"""
Sensitivity comparion
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
from load_root_hist import load_tb_j_pet_brain_insert_th3d, load_tb_j_pet_brain_insert_thnd


def main():
    # Output directory from the post-processing
    output_dir = '/home/martin/J-PET/Gate_Multi-detector_Post-processing/cmake-build-default/Output'

    # # Simulations with 200 ps CTR
    # tbtb_true, tbbi_true, bibi_true, z_edges = load_tb_j_pet_brain_insert_th3d(
    #     output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'true', return_grid=True)
    #
    # tbtb_energy, tbbi_energy, bibi_energy = load_tb_j_pet_brain_insert_th3d(
    #     output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'energy')
    # tbtb_energy_true, tbbi_energy_true, bibi_energy_true = load_tb_j_pet_brain_insert_th3d(
    #     output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'energy_true')
    #
    # tbtb_time, tbbi_time, bibi_time = load_tb_j_pet_brain_insert_th3d(
    #     output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'time')
    # tbtb_time_true, tbbi_time_true, bibi_time_true = load_tb_j_pet_brain_insert_th3d(
    #     output_dir + '/Sensitivity_line_source_200_ps_4_18_mm/2026-01-22_10-38-35_copy', 'time_true')

    # Simulation with 400 ps CTR
    tbtb_true, tbbi_true, bibi_true, z_edges = load_tb_j_pet_brain_insert_thnd(
        output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'true', return_grid=True)

    tbtb_energy, tbbi_energy, bibi_energy = load_tb_j_pet_brain_insert_thnd(
        output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'energy')
    tbtb_energy_true, tbbi_energy_true, bibi_energy_true = load_tb_j_pet_brain_insert_thnd(
        output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'energy_true')

    tbtb_time, tbbi_time, bibi_time = load_tb_j_pet_brain_insert_thnd(
        output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'time')
    tbtb_time_true, tbbi_time_true, bibi_time_true = load_tb_j_pet_brain_insert_thnd(
        output_dir + '/Sensitivity_line_source_400_ps_4_18_mm/2026-05-27_17-57-51', 'time_true')

    # Get the reference value
    n_simulations = 1
    run_time = 10000  # [s]
    activity = 1e6  # [Bq] = [1/s]
    source_length = 2540  # [mm]
    bin_width = z_edges[1] - z_edges[0]  # [mm]
    max_counts_per_bin = n_simulations * run_time * activity * bin_width / source_length
    max_counts_per_bin /= 1e3  # Rescale to cps / kBq

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.stairs((tbtb_energy_true + tbbi_energy_true + bibi_energy_true) / (tbtb_energy + tbbi_energy + bibi_energy) * 100, edges=z_edges, linestyle='-', color='tab:blue', label='ALL')
    ax.stairs(tbtb_energy_true / tbtb_energy * 100, edges=z_edges, linestyle='-', color='tab:orange', label='TB-TB')
    ax.stairs(tbbi_energy_true / tbbi_energy * 100, edges=z_edges, linestyle='-', color='tab:green', label='TB-BI')
    ax.stairs(bibi_energy_true / bibi_energy * 100, edges=z_edges, linestyle='-', color='tab:red', label='BI-BI')

    ax.stairs((tbtb_time_true + tbbi_time_true + bibi_time_true) / (tbtb_time + tbbi_time + bibi_time) * 100, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(tbtb_time_true / tbtb_time * 100, edges=z_edges, linestyle='--', color='tab:orange')
    ax.stairs(tbbi_time_true / tbbi_time * 100, edges=z_edges, linestyle='--', color='tab:green')
    ax.stairs(bibi_time_true / bibi_time * 100, edges=z_edges, linestyle='--', color='tab:red')

    ax.set_xlim([0, 1400])
    ax.set_ylim([65, 95])

    legend_1 = ax.legend(loc='lower center', frameon=True, ncol=4)
    h_dummy_1, = plt.plot(np.nan, color='black', linestyle='-')
    h_dummy_2, = plt.plot(np.nan, color='black', linestyle='--')
    ax.legend([h_dummy_1, h_dummy_2], ['Energy-based', 'Time-based'], frameon=False)
    ax.add_artist(legend_1)

    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel(r'Percentage of true events [%]')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
