"""
Sensitivity analysis based on Singles

@author: Martin Rädler
"""
# Python libraries
import sys

import matplotlib.pyplot as plt
import numpy as np
from uproot import open

# Auxiliary functions
# from Gate_9_0.sensitivity_analysis import get_coincidence_indices, get_sensitivity_profile, plot_sensitivity_profile, plot_distance_distribution
from Other.Gate_9_0.sensitivity_analysis import get_coincidence_indices, get_sensitivity_profile, plot_sensitivity_profile, plot_distance_distribution

def main():
    """
    Main function
    :return: 0
    """

    """Data simulated with Gate 9.0"""
    # Get the singles and construct a module index
    time_0, energy_0, source_pos_0, global_pos_0, event_id_0, photon_id_0, gantry_id_0, rsector_id_0, compton_crystal_0, rayleigh_crystal_0 = (
        read_singles_9_0('/home/martin/J-PET/Gate_mac_9.0/TB_J-PET_Brain_2/Output/2024-02-22_12-17-33/results.root'))
    module_id_0 = np.ravel_multi_index((rsector_id_0, gantry_id_0), (24, 2))

    # Get the coincidence indices
    idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton = (
        get_coincidence_indices(time_0, event_id_0, photon_id_0, module_id_0, compton_crystal_0, rayleigh_crystal_0, time_window=3e-9))

    # Get the sensitivity profile
    z_edges, z_centers, z_width, h_delta_t_0, h_event_id_photon_id_0, h_rsector_0, h_single_rayleigh_0, h_0, h_tbtb_0, h_bb_0, h_btb_0 = (
        get_sensitivity_profile(idx_sort, source_pos_0[:, 2], gantry_id_0,
                                idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh,
                                idx_1_single_compton, activity_kbq=1000., run_time_s=10.))

    # # Plot the sensitivity profile
    # plot_sensitivity_profile(z_edges, z_centers, z_width, h_delta_t_0, h_event_id_photon_id_0, h_rsector_0,
    #                          h_single_rayleigh_0, h_0, h_tbtb_0, h_btb_0, h_bb_0, y_lim=(0, 70))

    # Plot the distance distribution
    plot_distance_distribution(idx_sort, source_pos_0, global_pos_0, idx_1_single_compton, idx_1_multiple_compton)

    """Data simulated with Gate 9.3"""
    # Get the singles and construct a module index
    time_3, energy_3, source_pos_3, global_pos_3, event_id_3, photon_id_3, gantry_id_3, rsector_id_3, compton_crystal_3, rayleigh_crystal_3 = (
        read_singles_9_3('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/2024-02-22_11-54-44/results.root'))  #  (after fixing)  2024-02-22_11-49-02
    module_id_3 = np.ravel_multi_index((rsector_id_3, gantry_id_3), (24, 2))

    # Get the coincidence indices
    idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton = (
        get_coincidence_indices(time_3, event_id_3, photon_id_3, module_id_3, compton_crystal_3, rayleigh_crystal_3, time_window=3e-9))

    # Get the sensitivity profile
    _, _, _, h_delta_t_3, h_event_id_photon_id_3, h_rsector_3, h_single_rayleigh_3, h_3, h_tbtb_3, h_bb_3, h_btb_3 = (
        get_sensitivity_profile(idx_sort, source_pos_3[:, 2], gantry_id_3,
                                idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh,
                                idx_1_single_compton, activity_kbq=1000., run_time_s=10.))
    #
    # # Plot the sensitivity profile
    # plot_sensitivity_profile(z_edges, z_centers, z_width, h_delta_t_3, h_event_id_photon_id_3, h_rsector_3,
    #                          h_single_rayleigh_3, h_3, h_tbtb_3, h_btb_3, h_bb_3, y_lim=(0, 70))

    # Plot the distance distribution
    plot_distance_distribution(idx_sort, source_pos_3, global_pos_3, idx_1_single_compton, idx_1_multiple_compton)

    """Data analysis"""
    #
    interaction_statistics(compton_crystal_0, compton_crystal_3, 'comptonCrystal')
    interaction_statistics(rayleigh_crystal_0, rayleigh_crystal_3, 'RayleighCrystal')

    # Analyze the energy spectra
    energy_spectrum([energy_0, energy_3], ['Gate_9.0', 'Gate_9.3'], 'Singles: Compton counts')

    # Compare sensitivity profiles
    compare_sensitivities(z_edges, h_delta_t_0, h_0, h_delta_t_3, h_3)

    return 0


def read_singles_9_0(root_file_name):
    root_file = open(root_file_name)

    b_singles = root_file['B_Singles']
    singles_no_wls = root_file['SinglesNoWLS']
    time = np.concatenate((np.array(b_singles['time']), np.array(singles_no_wls['time'])))
    energy = np.concatenate((np.array(b_singles['energy']), np.array(singles_no_wls['energy'])))

    source_x = np.concatenate((np.array(b_singles['sourcePosX']), np.array(singles_no_wls['sourcePosX'])))
    source_y = np.concatenate((np.array(b_singles['sourcePosY']), np.array(singles_no_wls['sourcePosY'])))
    source_z = np.concatenate((np.array(b_singles['sourcePosZ']), np.array(singles_no_wls['sourcePosZ'])))
    source_pos = np.hstack((source_x[:, np.newaxis], source_y[:, np.newaxis], source_z[:, np.newaxis]))

    global_x = np.concatenate((np.array(b_singles['globalPosX']), np.array(singles_no_wls['globalPosX'])))
    global_y = np.concatenate((np.array(b_singles['globalPosY']), np.array(singles_no_wls['globalPosY'])))
    global_z = np.concatenate((np.array(b_singles['globalPosZ']), np.array(singles_no_wls['globalPosZ'])))
    global_pos = np.hstack((global_x[:, np.newaxis], global_y[:, np.newaxis], global_z[:, np.newaxis]))

    event_id = np.concatenate((np.array(b_singles['eventID']), np.array(singles_no_wls['eventID'])))
    photon_id = np.arange(event_id.size)  # dummy photon ID
    gantry_id = np.concatenate((np.array(b_singles['gantryID']), np.array(singles_no_wls['gantryID'])))
    rsector_id = np.concatenate((np.array(b_singles['rsectorID']), np.array(singles_no_wls['rsectorID'])))
    compton_crystal = np.concatenate((np.array(b_singles['comptonCrystal']), np.array(singles_no_wls['comptonCrystal'])))
    rayleigh_crystal = np.concatenate((np.array(b_singles['RayleighCrystal']), np.array(singles_no_wls['RayleighCrystal'])))

    return time, energy, source_pos, global_pos, event_id, photon_id, gantry_id, rsector_id, compton_crystal, rayleigh_crystal


def read_singles_9_3(root_file_name):
    root_file = open(root_file_name)
    merged_singles = root_file['MergedSingles_layer_2']

    time = np.array(merged_singles['time'])
    energy = np.array(merged_singles['energy'])

    source_x = np.array(merged_singles['sourcePosX'])
    source_y = np.array(merged_singles['sourcePosY'])
    source_z = np.array(merged_singles['sourcePosZ'])
    source_pos = np.hstack((source_x[:, np.newaxis], source_y[:, np.newaxis], source_z[:, np.newaxis]))

    global_x = np.array(merged_singles['globalPosX'])
    global_y = np.array(merged_singles['globalPosY'])
    global_z = np.array(merged_singles['globalPosZ'])
    global_pos = np.hstack((global_x[:, np.newaxis], global_y[:, np.newaxis], global_z[:, np.newaxis]))

    event_id = np.array(merged_singles['eventID'])
    photon_id = np.arange(event_id.size)  # dummy photon ID
    gantry_id = np.array(merged_singles['gantryID'])
    rsector_id = np.array(merged_singles['rsectorID'])
    compton_crystal = np.array(merged_singles['comptonCrystal'])
    rayleigh_crystal = np.array(merged_singles['RayleighCrystal'])

    return time, energy, source_pos, global_pos, event_id, photon_id, gantry_id, rsector_id, compton_crystal, rayleigh_crystal


def interaction_statistics(compton_crystal_0, compton_crystal_3, label):
    bc_0 = np.bincount(compton_crystal_0)
    bc_3 = np.bincount(compton_crystal_3)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(bc_0.size), bc_0, width=.9, alpha=0.5, label='Gate_9.0')
    ax.bar(np.arange(bc_3.size), bc_3, width=.9, alpha=0.5, label='Gate_9.3')

    ax.errorbar(np.arange(bc_0.size), bc_0, yerr=np.sqrt(bc_0), capsize=3, linestyle='none', color='tab:blue', alpha=0.5)
    ax.errorbar(np.arange(bc_3.size), bc_3, yerr=np.sqrt(bc_3), capsize=3, linestyle='none', color='tab:orange', alpha=0.5)

    x_ticks = np.arange(max(bc_0.size, bc_3.size))
    ax.set_xticks(x_ticks)

    x_tick_labels = x_ticks.astype(str)
    x_tick_labels[1::2] = ''
    ax.set_xticklabels(x_tick_labels)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax.set_yscale('log')
    ax.set_xlabel(label)
    ax.set_ylabel('Counts')
    ax.set_title('Singles: Compton counts')
    ax.legend()
    plt.show()
    return 0


def energy_spectrum(energies_list, labels_list, title=''):
    # Energy binning
    energy_bins = np.arange(512 + 1, dtype=float) / 1000
    # energy_bins = np.arange(300, 700 + 1, 10, dtype=float) / 1000
    energy_centers = (energy_bins[1:] + energy_bins[:-1]) / 2
    energy_widths = (energy_bins[1:] - energy_bins[:-1])

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    h_prev, _ = np.histogram(energies_list[0], bins=energy_bins)
    # ax.bar(energy_centers, h_prev, width=energy_widths, alpha=0.5, label=labels_list[0])
    ax.stairs(h_prev, edges=energy_bins, label=labels_list[0])

    for ii in range(1, len(energies_list)):
        # Get the histogram
        h_new, _ = np.histogram(energies_list[ii], bins=energy_bins)

        # Plot
        # ax.bar(energy_centers, h_new, width=energy_widths, alpha=0.5, label=labels_list[ii])
        ax.stairs(h_new, edges=energy_bins, label=labels_list[ii])

        # Scaling factor between
        alpha = np.sum(h_prev * h_new) / np.sum(h_prev * h_prev)
        print(alpha)

        h_prev = h_new.copy()


    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.legend()
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    plt.show()

    return 0


def compare_sensitivities(z_edges, h_0, h_0_filtered, h_3, h_3_filtered):

    alpha_03 = np.sum(h_0_filtered * h_3_filtered) / np.sum(h_3_filtered * h_3_filtered)
    print(alpha_03)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 6))
    s_0 = ax.stairs(h_0, edges=z_edges, color='tab:blue', label='Gate_9.0')
    s_3 = ax.stairs(h_3, edges=z_edges, color='tab:orange', label='Gate_9.3')

    ax.stairs(h_0_filtered, edges=z_edges, linestyle='--', color='tab:blue')
    ax.stairs(h_3_filtered, edges=z_edges, linestyle='--', color='tab:orange')
    # ax.stairs(h_3_filtered * alpha_03, edges=z_edges, linestyle=':', color='tab:orange', label=r'Gate_9.3 $\times$ %1.3f' % alpha_03)

    # Dummy plots for the legend
    first_legend = ax.legend(handles=[s_0, s_3], loc='upper center', frameon=True)

    d_0, = ax.plot(np.nan, color='black', label='Raw')
    d_3, = ax.plot(np.nan, linestyle='--', color='black', label='comptonCrystal == 1')
    ax.add_artist(first_legend)
    plt.legend(handles=[d_0, d_3], loc='lower center', frameon=True)
    ax.set_ylim(0, 70)
    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel('Sensitivity [cps/kBq]')
    ax.set_title('Sensitivity')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
