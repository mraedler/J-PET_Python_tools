"""
Analyze the sensitivity from Singles

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from re import finditer
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Auxiliary functions
from .tree_merger import layer_linear_to_subscript_indexing


def main():
    # Load the singles
    singles = np.load(sys.path[0] + '/data/singles_1.npy', allow_pickle=True)

    # Get an ID for the different modules
    module_id = np.ravel_multi_index((singles['rsectorID'], singles['gantryID']), (24, 2))

    # Get the coincidence indices
    idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton = (
        get_coincidence_indices(singles['time'], singles['eventID'], singles['photonID'], module_id,
                                singles['comptonCrystal'], singles['RayleighCrystal'], time_window=3e-9))

    # Get the sensitivity profile
    z_edges, z_centers, z_width, h_delta_t, h_event_id_photon_id, h_rsector, h_single_rayleigh, h, h_tbtb, h_bb, h_btb = (
        get_sensitivity_profile(idx_sort, singles['sourcePos'][:, 2], singles['gantryID'],
                                idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh,
                                idx_1_single_compton, activity_kbq=1000., run_time_s=10.))

    # # Plot the sensitivity profile
    # plot_sensitivity_profile(z_edges, z_centers, z_width, h_delta_t, h_event_id_photon_id, h_rsector, h_single_rayleigh,
    #                          h, h_tbtb, h_btb, h_bb)

    # Distance distribution
    plot_distance_distribution(idx_sort, singles['sourcePos'], singles['globalPos'], idx_1_single_compton, idx_1_multiple_compton)

    sys.exit()

    energy = singles['energy']

    layer_id_x, layer_id_y, layer_id_z = layer_linear_to_subscript_indexing(layer_id)

    # Both occurrences in the same module
    hs = (gantry_id[first_occurrence] == gantry_id[second_occurrence]) & (rsector_id[first_occurrence] == rsector_id[second_occurrence]) & (crystal_id[first_occurrence] == crystal_id[second_occurrence])

    a = np.abs(layer_id_x[first_occurrence][hs] - layer_id_x[second_occurrence][hs]) < 2
    b = np.abs(layer_id_y[first_occurrence][hs] - layer_id_y[second_occurrence][hs]) < 2
    c = np.abs(layer_id_z[first_occurrence][hs] - layer_id_z[second_occurrence][hs]) < 2

    print(np.sum(a & b & c))

    print(energy[first_occurrence][hs])

    print(np.sum(first_occurrence))


    # d = layer_id[]

    sys.exit()

    # print(layer_id_x)
    # print(layer_id_y)
    sys.exit()

    print(np.sum(first_occurrence) / time.size * 100)
    print(np.sum(first_occurrence & second_occurrence))


    # print(time[first_occurrence & second_occurrence])


    # fig, ax = plt.subplots()
    # ax.plot(singles['time'])
    # plt.show()

    return 0


def get_coincidence_indices(time, event_id, photon_id, module_id, compton_crystal, rayleigh_crystal, time_window=3e-9):
    # Sort the singles data according to time
    idx_sort = np.argsort(time)
    time = time[idx_sort]
    event_id = event_id[idx_sort]
    photon_id = photon_id[idx_sort]
    module_id = module_id[idx_sort]
    compton_crystal = compton_crystal[idx_sort]
    rayleigh_crystal = rayleigh_crystal[idx_sort]

    # Check the multiplicity
    coincidence_indices, coincidence_multiplicity = sort_multiplicity(time, time_window=time_window)

    # Indices of first (and second) occurrences
    time_diff_below_threshold = np.diff(time) < time_window
    first_occurrence = np.append(time_diff_below_threshold, False)
    # second_occurrence = np.insert(time_diff_below_threshold, 0, False)
    idx_1 = np.arange(time.size)[first_occurrence]

    # Boolean indices for various cuts
    event_id_photon_id = (event_id[idx_1] == event_id[idx_1 + 1]) & (photon_id[idx_1] != photon_id[idx_1 + 1])
    different_module = module_id[idx_1] != module_id[idx_1 + 1]
    no_rayleigh = (rayleigh_crystal[idx_1] == 0) & (rayleigh_crystal[idx_1 + 1] == 0)
    single_compton = (compton_crystal[idx_1] == 1) & (compton_crystal[idx_1 + 1] == 1)

    # Get the corresponding indices
    idx_1_delta_t = idx_1.copy()
    idx_1_event_id_photon_id = idx_1[event_id_photon_id]
    idx_1_different_modules = idx_1[event_id_photon_id & different_module]
    idx_1_no_rayleigh = idx_1[event_id_photon_id & different_module & no_rayleigh]
    idx_1_single_compton = idx_1[event_id_photon_id & different_module & no_rayleigh & single_compton]
    idx_1_multiple_compton = idx_1[event_id_photon_id & different_module & no_rayleigh & (~single_compton)]

    return idx_sort, idx_1_delta_t, idx_1_event_id_photon_id, idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton, idx_1_multiple_compton


def sort_multiplicity(time_sorted, time_window=3e-9):
    # Find coincidences
    time_diff_below_threshold = np.diff(time_sorted) < time_window
    n_coincidences = np.sum(time_diff_below_threshold)

    # Cast to string and attach False at the beginning and end
    time_diff_below_threshold_string = 'F' + ''.join('FT'[ii] for ii in time_diff_below_threshold.astype(int)) + 'F'

    # Allocate
    coincidence_indices = np.empty((0,), dtype=int)
    coincidence_multiplicity = np.empty((0,), dtype=int)
    multiplicity = 2
    total = 0
    pattern = '(?=FTF)'

    print('\nMultiplicity statistics\n=======================')
    while total < n_coincidences:
        # Find the indices of the given multiplicity
        indices = np.array([m.start() for m in finditer(pattern, time_diff_below_threshold_string)], dtype=int)
        n_found = indices.size
        total += n_found * (multiplicity - 1)
        print('%d: %d' % (multiplicity, n_found))

        # Append to array
        coincidence_indices = np.hstack((coincidence_indices, indices))
        coincidence_multiplicity = np.hstack((coincidence_multiplicity, multiplicity * np.ones(indices.shape, dtype=int)))

        # Increment the multiplicity and thus the pattern
        multiplicity += 1
        pattern = pattern[:4] + 'T' + pattern[4:]
    # Sort according to
    idx_sort = np.argsort(coincidence_indices)

    # # Consistency check (must be zero)
    # print(total - n_coincidences)

    return coincidence_indices[idx_sort], coincidence_multiplicity[idx_sort]


def get_sensitivity_profile(idx_sort, z_source, gantry_id, idx_1_delta_t, idx_1_event_id_photon_id,
                            idx_1_different_modules, idx_1_no_rayleigh, idx_1_single_compton,
                            activity_kbq=1000., run_time_s=10.):
    # Sort the data according to time
    z_source = z_source[idx_sort]
    gantry_id = gantry_id[idx_sort]

    # Histogram binning  todo: remove hard coding here
    n_bins = 61
    z_edges = np.linspace(-915, 915, n_bins + 1)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_width = z_edges[1:] - z_edges[:-1]

    # Non-separated histograms for various cuts
    h_delta_t, _ = np.histogram(z_source[idx_1_delta_t], bins=z_edges)
    h_event_id_photon_id, _ = np.histogram(z_source[idx_1_event_id_photon_id], bins=z_edges)
    h_modules, _ = np.histogram(z_source[idx_1_different_modules], bins=z_edges)
    h_single_rayleigh, _ = np.histogram(z_source[idx_1_no_rayleigh], bins=z_edges)

    # Separated histograms for the final cut
    z_data = z_source[idx_1_single_compton]
    gantry_id_first = gantry_id[idx_1_single_compton]
    gantry_id_second = gantry_id[idx_1_single_compton + 1]
    h, _ = np.histogram(z_data, bins=z_edges)
    h_tbtb, _ = np.histogram(z_data[(gantry_id_first == 0) & (gantry_id_second == 0)], bins=z_edges)
    h_bb, _ = np.histogram(z_data[(gantry_id_first == 1) & (gantry_id_second == 1)], bins=z_edges)
    h_btb, _ = np.histogram(z_data[gantry_id_first != gantry_id_second], bins=z_edges)

    # Normalization
    normalization = activity_kbq / n_bins * run_time_s
    h_delta_t = h_delta_t / normalization
    h_event_id_photon_id = h_event_id_photon_id / normalization
    h_modules = h_modules / normalization
    h_single_rayleigh = h_single_rayleigh / normalization

    h = h / normalization
    h_tbtb = h_tbtb / normalization
    h_bb = h_bb / normalization
    h_btb = h_btb / normalization

    return z_edges, z_centers, z_width, h_delta_t, h_event_id_photon_id, h_modules, h_single_rayleigh, h, h_tbtb, h_bb, h_btb


def plot_sensitivity_profile(z_edges, z_centers, z_width, h_delta_t, h_event_id_photon_id, h_rsector, h_single_rayleigh, h, h_tbtb, h_btb, h_bb, y_lim=None):
    # Figure
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.stairs(h_delta_t, edges=z_edges, color='k', linestyle='--', label=r'$\Delta t < 3$ ns')
    ax.stairs(h_event_id_photon_id, edges=z_edges, color='k', linestyle=':', label=r'eventID & photonID')
    ax.stairs(h_rsector, edges=z_edges, color='k', linestyle='-.', label=r'different rsectorID')
    ax.stairs(h_single_rayleigh, edges=z_edges, color='k', linestyle='-', label=r'nRayleigh = 0')

    ax.bar(z_centers, h, width=z_width, label='TOTAL')
    ax.bar(z_centers, h_tbtb, width=z_width, label='TB-TB')
    ax.bar(z_centers, h_btb, width=z_width, label='TB-B')
    ax.bar(z_centers, h_bb, width=z_width, label='B-B')
    ax.stairs(h, edges=z_edges, color='k', label=r'nCompton = 1', linewidth=2)

    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel('Sensitivity [cps/kBq]')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.legend(ncol=2)
    plt.show()
    return 0


def plot_distance_distribution(idx_sort, source_pos, global_pos, idx_1_single_compton, idx_1_multiple_compton):
    # Sort the data
    source_pos = source_pos[idx_sort]
    global_pos = global_pos[idx_sort]

    # Distances for both single Compton
    source_pos_single_0 = source_pos[idx_1_single_compton, :]
    # source_pos_single_1 = source_pos[idx_1_single_compton + 1, :]
    global_pos_single_0 = global_pos[idx_1_single_compton, :]
    global_pos_single_1 = global_pos[idx_1_single_compton + 1, :]
    d_single = minimum_distance(source_pos_single_0, global_pos_single_0, global_pos_single_1)
    # print('Distance exactly zero: %d' % np.sum(d_single == 0.))

    source_pos_multiple_0 = source_pos[idx_1_multiple_compton, :]
    # source_pos_multiple_1 = source_pos[idx_1_multiple_compton + 1, :]
    global_pos_multiple_0 = global_pos[idx_1_multiple_compton, :]
    global_pos_multiple_1 = global_pos[idx_1_multiple_compton + 1, :]
    d_multiple = minimum_distance(source_pos_multiple_0, global_pos_multiple_0, global_pos_multiple_1)
    # print('Distance exactly zero: %d' % np.sum(d_multiple == 0.))

    # Get the histogram
    distance_bin_edges = np.geomspace(1e-7, 1e3, 101)
    distance_bin_edges = np.insert(distance_bin_edges, 0, 0.)
    distance_bin_centers = (distance_bin_edges[1:] + distance_bin_edges[:-1]) / 2
    distance_bin_width = distance_bin_edges[1:] - distance_bin_edges[:-1]

    h_single, _ = np.histogram(d_single, bins=distance_bin_edges)
    h_multiple, _ = np.histogram(d_multiple, bins=distance_bin_edges)

    plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots(figsize=(16, 8))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(distance_bin_centers, h_single, width=distance_bin_width, alpha=0.75, label='nCompton = 1')
    ax.bar(distance_bin_centers, h_multiple, width=distance_bin_width, alpha=0.75, label='nCompton > 1')
    ax.set_xscale('symlog', linthresh=1e-7, linscale=0.1)
    # ax.set_yscale('log')
    ax.set_xlim(0, distance_bin_edges[-1])
    ax.set_xticks(ax.get_xticks()[1:])
    ax.set_xlabel('Minimum distance: LOR to source position [mm]')
    ax.set_ylabel('Counts')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.legend()
    plt.show()
    return 0


def minimum_distance(p, a, b):
    return np.linalg.norm(np.cross(p - a, b - a), axis=1) / np.linalg.norm(b - a, axis=1)


def singles_separation_with_identical_photon_id(singles):
    event_id = singles['eventID']
    photon_id = singles['photonID']
    global_pos = singles['globalPos']
    crystal_id = singles['crystalID']

    # Searching for singles with multiple entries
    sup_idx = np.ravel_multi_index((event_id, photon_id - 1), (np.max(event_id) + 1, 2))

    h = np.bincount(sup_idx)
    critical = np.arange(h.size)[h > 1]
    b = []
    d = []
    for ii in trange(critical.size):
        temp = global_pos[sup_idx == critical[ii], :]

        dist = np.linalg.norm(temp[1, :] - temp[0, :])

        b.append(temp)
        d.append(dist)

    # bin_edges = np.linspace(0, 1000, 81)
    bin_edges = np.geomspace(1, 2000, 81)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    hh, _ = np.histogram(np.array(d), bins=bin_edges)

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hh, width=bin_width)
    ax.set_xscale('log')
    plt.show()
    # print(b)

    # Extract the critical


    print(np.sum(h > 1))
    print(np.max(h))

    # if an event and photon number occurs twic

    #



    # # Determine eventIDs with multiplicity above 2
    # h = np.bincount(event_id)
    # event_id_min_3 = np.arange(h.size)[h > 2]
    # print(event_id_min_3.size)
    #
    # print(photon_id[event_id == event_id_min_3[0]])

    # Could also be among n=2


    print(np.bincount(np.bincount(event_id)))

    return 0


if __name__ == '__main__':
    main()
