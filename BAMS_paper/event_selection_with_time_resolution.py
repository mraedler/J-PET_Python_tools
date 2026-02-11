"""
Time or energy based event selection depending on the time resolution and energy threshold

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from tqdm import tqdm, trange
from time import sleep
import matplotlib.pyplot as plt

# Auxiliary functions
from utilities import load_gate_data, filter_true, filter_phantom_scattered, separate_into_detector_categories


def main():
    # lut = read_lut_binary(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert_dz_1_mm.lut')
    # lut_header = read_lut_header(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert_dz_1_mm.hscan')

    with_phantom = True
    # time_resolution = 0
    # time_resolution = 200
    # time_resolution = 400
    time_resolution = 600
    coincidences_struct = load_gate_data(time_resolution, with_phantom)

    # Preselection based on the minimum sector difference and the scatter test
    if with_phantom:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    coincidences_struct_filtered = coincidences_struct[preselection]

    # # Use only 10 %
    # coincidences_struct_filtered = coincidences_struct_filtered[:int(coincidences_struct_filtered.size * 1)]

    # Minimum distance
    if with_phantom:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    d_min_below_threshold = d_min <= 3.  # mm
    #
    true = filter_true(coincidences_struct_filtered)
    not_phantom_scattered = filter_phantom_scattered(coincidences_struct_filtered)

    # Get the multiplicity
    multiplicity, window_sizes, group_sizes, group_indices = get_multiplicity(coincidences_struct_filtered)
    multiplicity_histogram = multiplicity_analysis(multiplicity, window_sizes, group_sizes, group_indices, vis=False)
    # np.save(sys.path[0] + '/Multiplicity_plot/time_resolution_%d_ps_with_phantom.npy' % time_resolution, multiplicity_histogram)

    # Energy threshold-dependent time-based event selection
    energy_1, energy_2 = coincidences_struct_filtered['energy1'], coincidences_struct_filtered['energy2']
    energy_thresholds = np.linspace(50, 350, 20)[:, np.newaxis] * 1e-3  # MeV

    # Ideal event selection, assuming to choose one (or more) event per multiplicity group
    event_selection_ideal = ideal_event_selection(energy_thresholds, multiplicity, group_sizes, group_indices, true, not_phantom_scattered, energy_1, energy_2)

    event_selection_time_50_kev = time_event_selection_50_kev(group_indices, energy_1, energy_2, energy_thresholds)
    event_selection_time_variable = time_event_selection_variable(group_sizes, group_indices, energy_1, energy_2, energy_thresholds)

    # Energy-based event selection
    event_selection_energy_50_kev = energy_event_selection_50_kev(group_indices, energy_1, energy_2, energy_thresholds)
    event_selection_energy_variable = energy_event_selection_variable(group_indices, energy_1, energy_2, energy_thresholds)

    # Choosing one per multiplicity group
    n_events_id, p_true_id, p_phantom_scatter_id, p_phantom_scatter_forgiving_id = energy_threshold_variation_v2(event_selection_ideal, true, not_phantom_scattered, d_min_below_threshold)
    n_events_t5, p_true_t5, p_phantom_scatter_t5, p_phantom_scatter_forgiving_t5 = energy_threshold_variation_v2(event_selection_time_50_kev, true, not_phantom_scattered, d_min_below_threshold)
    n_events_tv, p_true_tv, p_phantom_scatter_tv, p_phantom_scatter_forgiving_tv = energy_threshold_variation_v2(event_selection_time_variable, true, not_phantom_scattered, d_min_below_threshold)
    n_events_e5, p_true_e5, p_phantom_scatter_e5, p_phantom_scatter_forgiving_e5 = energy_threshold_variation_v2(event_selection_energy_50_kev, true, not_phantom_scattered,d_min_below_threshold)
    n_events_ev, p_true_ev, p_phantom_scatter_ev, p_phantom_scatter_forgiving_ev = energy_threshold_variation_v2(event_selection_energy_variable, true, not_phantom_scattered, d_min_below_threshold)

    np.savez(sys.path[0] + '/Event_selection_Comb_plot/time_resolution_%d_ps_with_phantom.npz' % time_resolution, energy_thresholds=energy_thresholds.flatten(),
             n_events_id=n_events_id, p_true_id=p_true_id, p_phantom_scatter_id=p_phantom_scatter_id, p_phantom_scatter_forgiving_id=p_phantom_scatter_forgiving_id,
             n_events_t5=n_events_t5, p_true_t5=p_true_t5, p_phantom_scatter_t5=p_phantom_scatter_t5, p_phantom_scatter_forgiving_tv=p_phantom_scatter_forgiving_t5,
             n_events_tv=n_events_tv, p_true_tv=p_true_tv, p_phantom_scatter_tv=p_phantom_scatter_tv, p_phantom_scatter_forgiving_t5=p_phantom_scatter_forgiving_tv,
             n_events_e5=n_events_e5, p_true_e5=p_true_e5, p_phantom_scatter_e5=p_phantom_scatter_e5, p_phantom_scatter_forgiving_ev=p_phantom_scatter_forgiving_e5,
             n_events_ev=n_events_ev, p_true_ev=p_true_ev, p_phantom_scatter_ev=p_phantom_scatter_ev, p_phantom_scatter_forgiving_e5=p_phantom_scatter_forgiving_ev)

    np.savez(sys.path[0] + '/Event_selection/time_resolution_%d_ps_with_phantom.npz' % time_resolution,
             energy_thresholds=energy_thresholds.flatten(), ideal=event_selection_ideal,
             time_variable=event_selection_time_variable, time_50_kev=event_selection_time_50_kev,
             energy_variable=event_selection_energy_variable, energy_50_kev=event_selection_energy_50_kev,
             true=true, not_phantom_scattered=not_phantom_scattered)

    return 0


def get_multiplicity(coincidences_struct):
    # if one of the events was used multiple times in subsequent coincidences
    t1 = coincidences_struct['time1']
    t2 = coincidences_struct['time2']
    t12 = np.vstack((t1, t2))

    multiplicity = []
    window_sizes = []
    group_sizes = []
    ii = 0
    pbar = tqdm(total=coincidences_struct.size - 1)
    while ii < (coincidences_struct.size - 1):

        times = np.array([t1[ii], t2[ii]])
        coincidences_grouped = 1
        window_start = np.min(times)

        # Check subsequent coincidences for overlaps
        while ii < (coincidences_struct.size - 1):
            ii += 1
            times = np.append(times, [t1[ii], t2[ii]])
            times_size = times.size
            times = np.unique(times)

            # Check if either event is used in the subsequent coincidence, by checking if time values are repeated, i.e.
            # if np.unique(times) reduced the size
            if times.size < times_size:
                coincidences_grouped += 1
            else:
                break

        multiplicity.append(times.size - 2)
        window_stop = times[-3]
        window_sizes.append(window_stop - window_start)
        group_sizes.append(coincidences_grouped)

        pbar.update(coincidences_grouped)
    pbar.close()

    coincidences_processed = np.sum(group_sizes)
    if coincidences_processed == coincidences_struct.size:
        pass  # All coincidences were grouped
    elif coincidences_processed == (coincidences_struct.size - 1):
        # The last one is missing
        multiplicity.append(2)
        window_sizes.append(t2[-1] - t1[-1])
        group_sizes.append(1)
    else:
        sys.exit('Error: Not all coincidences were grouped.')

    #
    multiplicity = np.array(multiplicity)
    window_sizes = np.array(window_sizes)
    group_sizes = np.array(group_sizes)
    group_indices = np.insert(np.cumsum(group_sizes), 0, 0)

    # # Consistency check
    # for jj in trange(group_indices.size - 1):
    #     group = t12[:, group_indices[jj]:group_indices[jj + 1]]
    #     group_size = group.size
    #
    #     if group_size > 2:
    #         if np.unique(group).size == group_size:
    #             print(group)

    return multiplicity, window_sizes, group_sizes, group_indices


def multiplicity_analysis(multiplicity, window_sizes, group_sizes, group_indices, vis=False):
    multiplicity_histogram = np.bincount(multiplicity)

    bin_edges = np.linspace(0, 10e-9, 100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    window_size_histogram, _ = np.histogram(window_sizes, bins=bin_edges)

    if vis:
        plt.rcParams.update({'font.size': 16})
        fig, vis_ax = plt.subplots()
        x_multiplicity = np.arange(multiplicity_histogram.size)
        # y_histogram = multiplicity_histogram
        y_histogram = multiplicity_histogram / np.sum(multiplicity_histogram) * 100
        y_histogram[x_multiplicity < 2] = 0
        vis_ax.bar(x_multiplicity, y_histogram, width=0.8)
        # vis_ax.bar(bin_centers, window_size_histogram, width=bin_widths)
        vis_ax.set_xlim(np.min(x_multiplicity[y_histogram > 0]) - 1, np.max(x_multiplicity[y_histogram > 0]) + 1)

        vis_ax.set_xticks(x_multiplicity[y_histogram > 0])

        # vis_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        vis_ax.set_xlabel('Multiplicity')
        vis_ax.set_ylabel('Frequency [%]')

        ax_in = fig.add_axes([0.425, 0.425, 0.45, 0.425])
        ax_in.bar(x_multiplicity, y_histogram, width=0.8)
        ax_in.set_xlim(np.min(x_multiplicity[y_histogram > 0]) - 1, np.max(x_multiplicity[y_histogram > 0]) + 1)
        ax_in.set_xticks(x_multiplicity[y_histogram > 0][::2])
        ax_in.set_yscale('log')
        plt.show()
    return multiplicity_histogram


def ideal_event_selection(energy_thresholds, multiplicity, group_sizes, group_indices, true, not_phantom_scattered, energy_1, energy_2):
    above_energy_threshold = (energy_1 >= energy_thresholds) & (energy_2 >= energy_thresholds)

    # Ideal event selection
    event_selection = []
    for ii in trange(group_indices.size - 1):
        # group = np.arange(group_indices[ii], group_indices[ii + 1])
        # true_group = true[group]
        true_group = true[group_indices[ii]:group_indices[ii + 1]].copy()

        # if np.any(true_group):
        #     pass
        # else:
        #     true_group[0] = True

        if not np.any(true_group):
            true_group[0] = True

        event_selection.append(true_group)

    event_selection = np.hstack(event_selection)

    event_selection_2 = above_energy_threshold & event_selection






    # group_first_indices = group_indices[:-1]
    # least_one_event = np.maximum.reduceat(above_energy_threshold, group_first_indices, axis=1)
    # count = np.sum(least_one_event, axis=1)
    #
    # ll = np.sum(above_energy_threshold & true, axis=1)
    # kk = ll / count


    # print(np.sum(true[group_first_indices]) / group_first_indices.size)
    #
    # multiplicity_element_wise = np.repeat(multiplicity, group_sizes)
    #
    # for ii in range(2, np.max(multiplicity) + 1):
    #     p_true_first = np.sum(true[group_first_indices[multiplicity == ii]]) / np.sum(multiplicity == ii) * 100
    #     p_true = np.sum(true[multiplicity_element_wise == ii]) / np.sum(multiplicity == ii) * 100
    #
    #     print('%d: %1.2f < %1.2f %% (%1.2f)' % (ii, p_true_first, p_true, p_true_first / p_true * 100))

    return event_selection_2


def time_event_selection_50_kev(group_indices, energy_1, energy_2, energy_thresholds):
    above_energy_threshold = (energy_1 >= energy_thresholds) & (energy_2 >= energy_thresholds)

    group_firsts = np.zeros(energy_1.shape, dtype=bool)
    group_firsts[group_indices[:-1]] = True

    event_selection = above_energy_threshold & group_firsts

    return event_selection


def time_event_selection_variable(group_sizes, group_indices, energy_1, energy_2, energy_thresholds):
    above_energy_threshold = (energy_1 >= energy_thresholds) & (energy_2 >= energy_thresholds)

    selection = []

    for ii in trange(energy_thresholds.size):
        temp = []
        for jj in range(group_sizes.size):
            above_energy_threshold_group = above_energy_threshold[ii, group_indices[jj]:group_indices[jj + 1]]
            choice = np.zeros(group_sizes[jj], dtype=bool)
            if np.any(above_energy_threshold_group):
                choice[np.argmax(above_energy_threshold_group)] = True
            temp.append(choice)

        selection.append(np.concatenate(temp))

    selection = np.stack(selection)

    return selection


def energy_event_selection_50_kev(group_indices, energy_1, energy_2, energy_thresholds):
    above_energy_threshold = (energy_1 >= energy_thresholds) & (energy_2 >= energy_thresholds)

    energy_12 = np.vstack((energy_1, energy_2))
    energy_sum = energy_1 + energy_2

    event_selection = []
    for ii in trange(group_indices.size - 1):
        idx_group = np.arange(group_indices[ii], group_indices[ii + 1])
        idx_bool = np.zeros(idx_group.shape, dtype=bool)
        idx_bool[np.argmax(energy_sum[idx_group])] = True
        # event_selection[ii] = idx_group[np.argmax(energy_sum[idx_group])]
        event_selection.append(idx_bool)

        # e12_group = energy_12[:, idx_group]
        # candidates = e12_group == np.max(e12_group.flatten())
        # candidates = np.any(candidates, axis=0)
        # event_selection[ii] = idx_group[candidates][np.argmax(energy_sum[idx_group][candidates])]

    event_selection = np.hstack(event_selection)

    return event_selection & above_energy_threshold


def energy_event_selection_variable(group_indices, energy_1, energy_2, energy_thresholds):
    above_energy_threshold = (energy_1 >= energy_thresholds) & (energy_2 >= energy_thresholds)
    energy_sum = energy_1 + energy_2

    selection = []

    for ii in trange(energy_thresholds.size):
        temp = []
        for jj in range(group_indices.size - 1):
            above_energy_threshold_group = above_energy_threshold[ii, group_indices[jj]:group_indices[jj + 1]]
            energy_sum_group = energy_sum[group_indices[jj]:group_indices[jj + 1]]
            choice = np.zeros(above_energy_threshold_group.size, dtype=bool)
            if np.any(above_energy_threshold_group):
                idx = np.argmax(energy_sum_group[above_energy_threshold_group])
                ttt = np.zeros(np.sum(above_energy_threshold_group), dtype=bool)
                ttt[idx] = True
                choice[above_energy_threshold_group] = ttt

            temp.append(choice)

        selection.append(np.concatenate(temp))

    selection = np.stack(selection)

    return selection


def energy_threshold_variation_v2(event_selection, true, not_phantom_scattered, d_min_below_threshold):

    not_phantom_scattered_forgiving = not_phantom_scattered | d_min_below_threshold

    n_events = np.sum(event_selection, axis=1)
    p_true = np.sum(event_selection & true[np.newaxis, :], axis=1) / n_events * 100
    p_phantom_scatter = np.sum(event_selection & not_phantom_scattered[np.newaxis, :], axis=1) / n_events * 100
    p_phantom_scatter_forgiving = np.sum(event_selection & not_phantom_scattered_forgiving[np.newaxis, :], axis=1) / n_events * 100
    return n_events, p_true, p_phantom_scatter, p_phantom_scatter_forgiving


def event_selection_detector_categorized():

    with_phantom = True
    # time_resolution = 0
    # time_resolution = 200
    # time_resolution = 400
    time_resolution = 600
    coincidences_struct = load_gate_data(time_resolution, with_phantom)

    # Preselection based on the minimum sector difference and the scatter test
    if with_phantom:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    coincidences_struct_filtered = coincidences_struct[preselection]

    # Minimum distance
    if with_phantom:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    d_min_below_threshold = d_min <= 3.  # mm

    if with_phantom:
        npz_file = np.load(sys.path[0] + '/Event_selection/time_resolution_%d_ps_with_phantom.npz' % time_resolution)
    else:
        npz_file = np.load(sys.path[0] + '/Event_selection/time_resolution_%d_ps_without_phantom.npz' % time_resolution)

    energy_thresholds = npz_file['energy_thresholds']
    event_selection_ideal = npz_file['ideal']
    event_selection_time_variable = npz_file['time_variable']
    event_selection_time_50_kev = npz_file['time_50_kev']
    event_selection_energy_variable = npz_file['energy_variable']
    event_selection_energy_50_kev = npz_file['energy_50_kev']
    true = npz_file['true']
    not_phantom_scattered = npz_file['not_phantom_scattered']

    tbtb, tbbi, bibi = separate_into_detector_categories(coincidences_struct_filtered['gantryID1'], coincidences_struct_filtered['gantryID2'], verbose=True)
    sys.exit()

    # detector_category = np.ones(coincidences_struct_filtered.size, dtype=bool)
    detector_category = bibi.copy()
    # detector_category = tbbi.copy()

    event_selection_ideal = event_selection_ideal[:, detector_category]
    event_selection_time_variable = event_selection_time_variable[:, detector_category]
    event_selection_time_50_kev = event_selection_time_50_kev[:, detector_category]
    event_selection_energy_variable = event_selection_energy_variable[:, detector_category]
    event_selection_energy_50_kev = event_selection_energy_50_kev[:, detector_category]
    true = true[detector_category]
    not_phantom_scattered = not_phantom_scattered[detector_category]
    d_min_below_threshold = d_min_below_threshold[detector_category]

    # Choosing one per multiplicity group
    n_events_id, p_true_id, p_phantom_scatter_id, p_phantom_scatter_forgiving_id = energy_threshold_variation_v2(event_selection_ideal, true, not_phantom_scattered, d_min_below_threshold)
    n_events_t5, p_true_t5, p_phantom_scatter_t5, p_phantom_scatter_forgiving_t5 = energy_threshold_variation_v2(event_selection_time_50_kev, true, not_phantom_scattered, d_min_below_threshold)
    n_events_tv, p_true_tv, p_phantom_scatter_tv, p_phantom_scatter_forgiving_tv = energy_threshold_variation_v2(event_selection_time_variable, true, not_phantom_scattered, d_min_below_threshold)
    n_events_e5, p_true_e5, p_phantom_scatter_e5, p_phantom_scatter_forgiving_e5 = energy_threshold_variation_v2(event_selection_energy_50_kev, true, not_phantom_scattered,d_min_below_threshold)
    n_events_ev, p_true_ev, p_phantom_scatter_ev, p_phantom_scatter_forgiving_ev = energy_threshold_variation_v2(event_selection_energy_variable, true, not_phantom_scattered, d_min_below_threshold)

    np.savez(sys.path[0] + '/Event_selection_BI-BI_plot/time_resolution_%d_ps_without_phantom.npz' % time_resolution, energy_thresholds=energy_thresholds.flatten(),
             n_events_id=n_events_id, p_true_id=p_true_id, p_phantom_scatter_id=p_phantom_scatter_id, p_phantom_scatter_forgiving_id=p_phantom_scatter_forgiving_id,
             n_events_t5=n_events_t5, p_true_t5=p_true_t5, p_phantom_scatter_t5=p_phantom_scatter_t5, p_phantom_scatter_forgiving_tv=p_phantom_scatter_forgiving_t5,
             n_events_tv=n_events_tv, p_true_tv=p_true_tv, p_phantom_scatter_tv=p_phantom_scatter_tv, p_phantom_scatter_forgiving_t5=p_phantom_scatter_forgiving_tv,
             n_events_e5=n_events_e5, p_true_e5=p_true_e5, p_phantom_scatter_e5=p_phantom_scatter_e5, p_phantom_scatter_forgiving_ev=p_phantom_scatter_forgiving_e5,
             n_events_ev=n_events_ev, p_true_ev=p_true_ev, p_phantom_scatter_ev=p_phantom_scatter_ev, p_phantom_scatter_forgiving_e5=p_phantom_scatter_forgiving_ev)

    return 0


def plot_event_selection():
    time_resolution = ['0', '200', '400', '600']
    line_styles = ['-', '-.', '--', ':']

    detector_category = 'Comb'
    # detector_category = 'BI-BI'
    # detector_category = 'TB-BI'

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=(2, 3), figsize=(8, 8))
    for ii in range(len(time_resolution)):
        npz_file = np.load(sys.path[0] + '/Event_selection_%s_plot/time_resolution_%s_ps_without_phantom.npz' % (detector_category, time_resolution[ii]))
        # npz_file = np.load(sys.path[0] + '/Event_selection_%s_plot/time_resolution_%s_ps_with_phantom.npz' % (detector_category, time_resolution[ii]))

        energy_thresholds = npz_file['energy_thresholds']
        n_events_id, p_true_id, p_phantom_scatter_id, p_phantom_scatter_forgiving_id = npz_file['n_events_id'], npz_file['p_true_id'], npz_file['p_phantom_scatter_id'], npz_file['p_phantom_scatter_forgiving_id']
        n_events_t5, p_true_t5, p_phantom_scatter_t5, p_phantom_scatter_forgiving_t5 = npz_file['n_events_t5'], npz_file['p_true_t5'], npz_file['p_phantom_scatter_t5'], npz_file['p_phantom_scatter_forgiving_t5']
        n_events_tv, p_true_tv, p_phantom_scatter_tv, p_phantom_scatter_forgiving_tv = npz_file['n_events_tv'], npz_file['p_true_tv'], npz_file['p_phantom_scatter_tv'], npz_file['p_phantom_scatter_forgiving_tv']
        n_events_e5, p_true_e5, p_phantom_scatter_e5, p_phantom_scatter_forgiving_e5 = npz_file['n_events_e5'], npz_file['p_true_e5'], npz_file['p_phantom_scatter_e5'], npz_file['p_phantom_scatter_forgiving_e5']
        # n_events_ev, p_true_ev, p_phantom_scatter_ev, p_phantom_scatter_forgiving_ev = npz_file['n_events_ev'], npz_file['p_true_ev'], npz_file['p_phantom_scatter_ev'], npz_file['p_phantom_scatter_forgiving_ev']

        reference_id = np.interp(200e-3, energy_thresholds, n_events_id)
        reference_t5 = np.interp(200e-3, energy_thresholds, n_events_t5)
        reference_tv = np.interp(200e-3, energy_thresholds, n_events_tv)
        reference_e5 = np.interp(200e-3, energy_thresholds, n_events_e5)

        ax0.plot(energy_thresholds * 1e3, n_events_id / reference_id, color='tab:blue', linestyle=line_styles[ii])
        ax0.plot(energy_thresholds * 1e3, n_events_t5 / reference_t5, color='tab:orange', linestyle=line_styles[ii])
        ax0.plot(energy_thresholds * 1e3, n_events_tv / reference_tv, color='tab:green', linestyle=line_styles[ii])
        ax0.plot(energy_thresholds * 1e3, n_events_e5 / reference_e5, color='tab:red', linestyle=line_styles[ii])

        ax1.plot(energy_thresholds * 1e3, p_true_id, color='tab:blue', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, p_true_t5, color='tab:orange', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, p_true_tv, color='tab:green', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, p_true_e5, color='tab:red', linestyle=line_styles[ii])

        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_id, color='tab:blue', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_t5, color='tab:orange', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_tv, color='tab:green', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_e5, color='tab:red', linestyle=line_styles[ii])
        #
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_forgiving_id, color='tab:blue', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_forgiving_t5, color='tab:orange', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_forgiving_tv, color='tab:green', linestyle=line_styles[ii])
        # ax1.plot(energy_thresholds * 1e3, p_phantom_scatter_forgiving_e5, color='tab:red', linestyle=line_styles[ii])

    x_lim = ax0.get_xlim()
    y_lim = ax0.get_ylim()
    ax0.plot([200, 200], [y_lim[0], 1], color='k', linestyle='--')
    ax0.plot([x_lim[0], 200], [1, 1], color='k', linestyle='--')
    ax0.set_ylim(y_lim)

    p0, = ax0.plot(np.nan, color='k', linestyle='-')
    p1, = ax0.plot(np.nan, color='k', linestyle='-.')
    p2, = ax0.plot(np.nan, color='k', linestyle='--')
    p3, = ax0.plot(np.nan, color='k', linestyle=':')
    legend = ax0.legend(handles=[p0, p1, p2, p3], labels=time_resolution,
                        loc='upper right', frameon=True, title=r'$\bf{FWHM}$' '\n' r'$\mathbf{\Delta t\,[ps]}$')

    ax0.set_xlim(35, 365)
    ax0.set_ylabel(r'$N_c(E)\,/\,N_c(200\,\mathrm{keV})$')
    ax0.grid()

    p1, = ax1.plot(np.nan, color='tab:blue')
    p2, = ax1.plot(np.nan, color='tab:orange')
    p3, = ax1.plot(np.nan, color='tab:green')
    p4, = ax1.plot(np.nan, color='tab:red')
    # ax1.legend(handles=[p1, p2, p3, p4], labels=['Ideal', 'Time-based\nat 50 keV', 'Time-based\nvariable', 'Energy-based'],
    #            loc='lower right', frameon=False, title=r'$\bf{Event\,\,selection}$', ncol=1)
    ax1.legend(handles=[p1, p2, p3, p4], labels=['Ideal', 'Time-based\nat 50 keV', 'Time-based\nvariable', 'Energy-based'],
               loc='lower right', frameon=False, title=r'$\bf{Event}$' '\n' r'$\bf{selection}$', ncol=1)

    # ax0.add_artist(legend)

    # ax1.annotate('True in detector', xy=(90, 81), xytext=(90, 90), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom', xy=(175, 25), xytext=(175, 12), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom\n' r'$\pm3$ mm tol.', xy=(175, 41), xytext=(175, 51), ha='center', arrowprops=dict(arrowstyle='->'))

    # ax1.annotate('True in detector', xy=(90, 88), xytext=(90, 97), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom', xy=(175, 25), xytext=(175, 12), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom\n' r'$\pm3$ mm tol.', xy=(175, 56), xytext=(175, 32), ha='center', arrowprops=dict(arrowstyle='->'))

    # ax1.annotate('True in detector', xy=(90, 81), xytext=(90, 90), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom', xy=(175, 28), xytext=(175, 15), ha='center', arrowprops=dict(arrowstyle='->'))
    # ax1.annotate('True in phantom\n' r'$\pm3$ mm tol.', xy=(175, 52), xytext=(175, 32), ha='center', arrowprops=dict(arrowstyle='->'))

    ax1.set_xlim(35, 365)
    ax1.set_xlabel(r'Energy threshold $E$ [keV]')
    # ax1.set_ylabel('Percentage of true events [%]')
    ax1.set_ylabel('Percentage [%]')
    # ax1.set_ylim(65 - 1, 95 + 1)

    plt.show()

    return 0


def plot_multiplicity():
    time_resolution = ['0', '200', '400', '600']
    line_styles = ['-', '-.', '--', ':']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax_in = fig.add_axes([0.425, 0.425, 0.45, 0.425])
    max_x = 0
    for ii in range(len(time_resolution)):
        mult_hist = np.load(sys.path[0] + '/Multiplicity_plot/time_resolution_%s_ps_without_phantom.npy' % (time_resolution[ii]))
        mult_hist_phantom = np.load(sys.path[0] + '/Multiplicity_plot/time_resolution_%s_ps_with_phantom.npy' % (time_resolution[ii]))

        # print(mult_hist[2:-1] / mult_hist[3:])
        print(mult_hist_phantom[2:-1] / mult_hist_phantom[3:])
        # sys.exit()

        mult = np.arange(mult_hist.size)
        mult_phantom = np.arange(mult_hist_phantom.size)

        if mult[-1] > max_x:
            max_x = mult[-1]

        mult_hist = mult_hist / np.sum(mult_hist) * 100
        mult_hist[mult < 2] = 0

        mult_hist_phantom = mult_hist_phantom / np.sum(mult_hist_phantom) * 100
        mult_hist_phantom[mult_phantom < 2] = 0

        ax.bar(mult, mult_hist, width=1, color='none', edgecolor='tab:blue', linestyle=line_styles[ii])
        ax.bar(mult_phantom, mult_hist_phantom, width=1, color='none', edgecolor='tab:orange', linestyle=line_styles[ii])

        ax_in.bar(mult, mult_hist, width=1, color='none', edgecolor='tab:blue', linestyle=line_styles[ii])
        ax_in.bar(mult_phantom, mult_hist_phantom, width=1, color='none', edgecolor='tab:orange', linestyle=line_styles[ii])

    ax.set_xlim(1, max_x + 1)
    ax.set_xticks(np.arange(2, max_x + 1))
    ax.set_xlabel('Multiplicity')
    ax.set_ylabel('Frequency [%]')

    p0, = ax.plot(np.nan, color='k', linestyle='-')
    p1, = ax.plot(np.nan, color='k', linestyle='-.')
    p2, = ax.plot(np.nan, color='k', linestyle='--')
    p3, = ax.plot(np.nan, color='k', linestyle=':')
    ax.legend(handles=[p0, p1, p2, p3], labels=time_resolution, loc='lower right', frameon=False, ncol=2, title=r'$\bf{FWHM}$ $\mathbf{\Delta t\,[ps]}$')

    ax_in.set_xlim(1, max_x + 1)
    ax_in.set_xticks(np.arange(2, max_x + 1)[::2])
    ax_in.set_yscale('log')

    p0, = ax_in.plot(np.nan, color='tab:blue')
    p1, = ax_in.plot(np.nan, color='tab:orange')
    legend = ax_in.legend(handles=[p0, p1], labels=['w/o', 'w/'], loc='upper right', frameon=False, title=r'$\bf{Phantom}$')

    plt.show()
    return 0


if __name__ == "__main__":
    # main()
    # event_selection_detector_categorized()
    plot_event_selection()
    # plot_multiplicity()

