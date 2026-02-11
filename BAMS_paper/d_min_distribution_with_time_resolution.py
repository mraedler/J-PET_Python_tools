"""
Analyzing the d_min distribution depending on the time resolution and energy threshold

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import BoundaryNorm

# Auxiliary functions
from utilities import load_gate_data


def main():
    with_phantom = False
    # time_resolution = 0
    # time_resolution = 200
    # time_resolution = 400
    time_resolution = 600
    coincidences_struct = load_gate_data(time_resolution, with_phantom)

    print(coincidences_struct['time1'][-1])
    print(coincidences_struct['time2'][-1])
    sys.exit()

    # Preselection based on the minimum sector difference and the scatter test
    if with_phantom:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        preselection = np.load(sys.path[0] + '/Preselection/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    coincidences_struct_filtered = coincidences_struct[preselection]

    # Load the event selection
    if with_phantom:
        event_selection = np.load(sys.path[0] + '/Event_selection/time_resolution_%d_ps_with_phantom.npz' % time_resolution)
    else:
        event_selection = np.load(sys.path[0] + '/Event_selection/time_resolution_%d_ps_without_phantom.npz' % time_resolution)


    # d_min = get_d_min(coincidences_struct_filtered)
    # np.save(sys.path[0] + '/d_min/time_resolution_0_ps_without_phantom.npy', d_min)
    # Load d_min
    if with_phantom:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_with_phantom.npy' % time_resolution)
    else:
        d_min = np.load(sys.path[0] + '/d_min/time_resolution_%d_ps_without_phantom.npy' % time_resolution)

    energy_thresholds = event_selection['energy_thresholds']
    ideal = event_selection['ideal']
    time_variable = event_selection['time_variable']
    time_50_kev = event_selection['time_50_kev']
    energy_variable = event_selection['energy_variable']
    energy_50_kev = event_selection['energy_50_kev']

    true = event_selection['true']
    not_phantom_scattered = event_selection['not_phantom_scattered']

    same_event_ids = coincidences_struct_filtered['eventID1'] == coincidences_struct_filtered['eventID2']
    # selection = same_event_ids
    selection = same_event_ids & (~ not_phantom_scattered) & true

    # Use only 10 %
    # coincidences_struct_filtered = coincidences_struct_filtered[:int(coincidences_struct_filtered.size * .1)]

    d_bin_edges = np.logspace(-4, 4, 100)
    d_bin_edges = np.insert(d_bin_edges, 0, 0)

    h_ideal = lor_source_point_distance_distribution(d_min, energy_thresholds, ideal, selection, d_bin_edges)
    h_time_variable = lor_source_point_distance_distribution(d_min, energy_thresholds, time_variable, selection, d_bin_edges)
    h_time_50_kev = lor_source_point_distance_distribution(d_min, energy_thresholds, time_50_kev, selection, d_bin_edges)
    h_energy_variable = lor_source_point_distance_distribution(d_min, energy_thresholds, energy_variable, selection, d_bin_edges)
    h_energy_50_kev = lor_source_point_distance_distribution(d_min, energy_thresholds, energy_50_kev, selection, d_bin_edges)

    np.savez(sys.path[0] + '/d_min_distributions/time_resolution_%d_ps_with_phantom.npz' % time_resolution,
             energy_thresholds=energy_thresholds, d_bin_edges=d_bin_edges, h_ideal=h_ideal,
             h_time_variable=h_time_variable, h_time_50_kev=h_time_50_kev,
             h_energy_variable=h_energy_variable, h_energy_50_kev=h_energy_50_kev)

    return 0


def lor_source_point_distance_distribution(d_min, energy_thresholds, event_selection, selection, d_bin_edges):
    # source_pos = np.stack((coincidences_struct['sourcePosX1'],
    #                        coincidences_struct['sourcePosY1'],
    #                        coincidences_struct['sourcePosZ1']), axis=1)
    #
    # global_pos_1 = np.stack((coincidences_struct['globalPosX1'],
    #                          coincidences_struct['globalPosY1'],
    #                          coincidences_struct['globalPosZ1']), axis=1)
    #
    # global_pos_2 = np.stack((coincidences_struct['globalPosX2'],
    #                          coincidences_struct['globalPosY2'],
    #                          coincidences_struct['globalPosZ2']), axis=1)
    #
    # d_min = np.linalg.norm(np.cross(source_pos - global_pos_1, global_pos_2 - global_pos_1, axis=1), axis=1) / np.linalg.norm(global_pos_2 - global_pos_1, axis=1)

    d_min_idx = np.digitize(d_min, bins=d_bin_edges) - 1

    h_2d = np.zeros((energy_thresholds.size, d_bin_edges.size - 1))

    for ii in trange(energy_thresholds.size):
        # d_min_idx_above_energy_threshold = d_min_idx[event_selection[ii, :] & same_event_ids]
        d_min_idx_above_energy_threshold = d_min_idx[event_selection[ii, :] & selection]
        h_2d[ii, :] = np.bincount(d_min_idx_above_energy_threshold, minlength=d_bin_edges.size - 1)
        # h_2d[ii, :] = np.bincount(d_min_idx_above_energy_threshold[d_min_idx_above_energy_threshold >= 0], minlength=d_bin_centers.size)

    return h_2d


def get_d_min(coincidences_struct):
    source_pos = np.stack((coincidences_struct['sourcePosX1'],
                           coincidences_struct['sourcePosY1'],
                           coincidences_struct['sourcePosZ1']), axis=1)

    global_pos_1 = np.stack((coincidences_struct['globalPosX1'],
                             coincidences_struct['globalPosY1'],
                             coincidences_struct['globalPosZ1']), axis=1)

    global_pos_2 = np.stack((coincidences_struct['globalPosX2'],
                             coincidences_struct['globalPosY2'],
                             coincidences_struct['globalPosZ2']), axis=1)

    d_min = np.linalg.norm(np.cross(source_pos - global_pos_1, global_pos_2 - global_pos_1, axis=1), axis=1) / np.linalg.norm(global_pos_2 - global_pos_1, axis=1)

    return d_min


def plot_d_min_distributions():
    time_resolution = ['0', '200', '400', '600']
    line_styles = ['-', '-.', '--', ':']

    """Median plot"""
    plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots()
    fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=(6.5, 7))
    for ii in range(len(time_resolution)):
        # npz_file = np.load(sys.path[0] + '/d_min_distributions/time_resolution_%s_ps_without_phantom.npz' % time_resolution[ii])
        npz_file = np.load(sys.path[0] + '/d_min_distributions/time_resolution_%s_ps_with_phantom.npz' % time_resolution[ii])

        energy_thresholds = npz_file['energy_thresholds']
        d_bin_edges = npz_file['d_bin_edges']
        d_bin_centers = (d_bin_edges[:-1] + d_bin_edges[1:]) / 2

        h_ideal = npz_file['h_ideal']
        h_time_50_kev = npz_file['h_time_50_kev']
        h_time_variable = npz_file['h_time_variable']
        h_energy_50_kev = npz_file['h_energy_50_kev']
        # h_energy_variable = npz_file['h_energy_variable']

        # # Frequency of good events
        # ax.plot(energy_thresholds, h_ideal[:, 0] / np.sum(h_ideal, axis=1), color='tab:blue', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds, h_time_50_kev[:, 0] / np.sum(h_time_50_kev, axis=1), color='tab:orange', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds, h_time_variable[:, 0] / np.sum(h_time_variable, axis=1), color='tab:green', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds, h_energy_50_kev[:, 0] / np.sum(h_energy_50_kev, axis=1), color='tab:red', linestyle=line_styles[ii])

        # Median of the d_min distribution
        # ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges[1:], h_ideal[:, 1:]), color='tab:blue', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges[1:], h_time_50_kev[:, 1:]), color='tab:orange', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges[1:], h_time_variable[:, 1:]), color='tab:green', linestyle=line_styles[ii])
        # ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges[1:], h_energy_50_kev[:, 1:]), color='tab:red', linestyle=line_styles[ii])
        ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges, h_ideal), color='tab:blue', linestyle=line_styles[ii])
        ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges, h_time_50_kev), color='tab:orange', linestyle=line_styles[ii])
        ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges, h_time_variable), color='tab:green', linestyle=line_styles[ii])
        ax.plot(energy_thresholds * 1e3, get_median(d_bin_edges, h_energy_50_kev), color='tab:red', linestyle=line_styles[ii])

        #
        mm_tresh = 6.
        ax1.plot(energy_thresholds * 1e3, np.sum(h_ideal[:, d_bin_centers < mm_tresh], axis=1), color='tab:blue', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, np.sum(h_time_50_kev[:, d_bin_centers < mm_tresh], axis=1), color='tab:orange', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, np.sum(h_time_variable[:, d_bin_centers < mm_tresh], axis=1), color='tab:green', linestyle=line_styles[ii])
        ax1.plot(energy_thresholds * 1e3, np.sum(h_energy_50_kev[:, d_bin_centers < mm_tresh], axis=1), color='tab:red', linestyle=line_styles[ii])

        ax2.plot(energy_thresholds * 1e3, 100 * np.sum(h_ideal[:, d_bin_centers < mm_tresh], axis=1) / np.sum(h_ideal, axis=1), color='tab:blue', linestyle=line_styles[ii])
        ax2.plot(energy_thresholds * 1e3, 100 * np.sum(h_time_50_kev[:, d_bin_centers < mm_tresh], axis=1) / np.sum(h_time_50_kev, axis=1), color='tab:orange', linestyle=line_styles[ii])
        ax2.plot(energy_thresholds * 1e3, 100 * np.sum(h_time_variable[:, d_bin_centers < mm_tresh], axis=1) / np.sum(h_time_variable, axis=1), color='tab:green', linestyle=line_styles[ii])
        ax2.plot(energy_thresholds * 1e3, 100 * np.sum(h_energy_50_kev[:, d_bin_centers < mm_tresh], axis=1) / np.sum(h_energy_50_kev, axis=1), color='tab:red', linestyle=line_styles[ii])

    # ax.set_xlabel('Energy threshold [keV]')
    # ax.set_ylabel(r'Median [mm] of $d_\mathrm{min}\geq10^{-4}$ mm')

    ax.set_ylabel('Median [mm]\n' r'of $d_\mathrm{min}$')
    ax.set_xticklabels([])
    ax.set_yticks([0, 50, 100, 150])

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_xticklabels([])
    # ax1.set_ylabel(r'$N_c$ of $d_\mathrm{min} < 6$ mm')
    ax1.set_ylabel(r'$N_c$ [counts]' '\n' r'of $d_\mathrm{min} < 6$ mm')

    ax2.set_xlabel('Energy threshold [keV]')
    ax2.set_ylabel('Percentage [%]\n' r'of $d_\mathrm{min} < 6$ mm')











    p0, = ax.plot(np.nan, color='k', linestyle='-')
    p1, = ax.plot(np.nan, color='k', linestyle='-.')
    p2, = ax.plot(np.nan, color='k', linestyle='--')
    p3, = ax.plot(np.nan, color='k', linestyle=':')
    # legend = ax.legend(handles=[p0, p1, p2, p3], labels=time_resolution,
    #                    loc='lower left', frameon=False, title=r'$\bf{FWHM}$' '\n' r'$\mathbf{\Delta t\,[ps]}$', ncol=2)
    legend = ax.legend(handles=[p0, p1, p2, p3], labels=time_resolution,
                       loc='lower left', frameon=False, title=r'$\bf{FWHM}\,\,\mathbf{\Delta t\,[ps]}$', ncol=2, fontsize=12)

    p1, = ax.plot(np.nan, color='tab:blue')
    p2, = ax.plot(np.nan, color='tab:orange')
    p3, = ax.plot(np.nan, color='tab:green')
    p4, = ax.plot(np.nan, color='tab:red')
    # ax.legend(handles=[p1, p2, p3, p4], labels=['Ideal', 'Time-based\nat 50 keV', 'Time-based\nvariable', 'Energy-based'],
    #           loc='upper right', frameon=False, title=r'$\bf{Event\,\,selection}$', ncol=1)
    ax.legend(handles=[p1, p2, p3, p4], labels=['Ideal', 'Time-based at 50 keV', 'Time-based variable', 'Energy-based'],
              loc='upper right', frameon=False, title=r'$\bf{Event\,\,selection}$', ncol=1, fontsize=12)

    ax.add_artist(legend)
    plt.show()
    # plt.close()

    """Distributions"""

    npz_file = np.load(sys.path[0] + '/d_min_distributions/time_resolution_0_ps_without_phantom.npz')
    # npz_file = np.load(sys.path[0] + '/d_min_distributions/time_resolution_0_ps_with_phantom.npz')
    energy_thresholds = npz_file['energy_thresholds']
    d_bin_edges = npz_file['d_bin_edges']
    h_ideal = npz_file['h_ideal']
    h_time_50_kev = npz_file['h_time_50_kev']
    h_time_variable = npz_file['h_time_variable']
    h_energy_50_kev = npz_file['h_energy_50_kev']

    plt.rcParams.update({'font.size': 16})
    fig, ((ax00, ax01, axc0), (ax10, ax11, axc1)) = plt.subplots(2, 3, figsize=(8, 6), width_ratios=(1, 1, 0.05))
    c_map = get_cmap('viridis', energy_thresholds.size)

    m_ideal = get_median(d_bin_edges[1:], h_ideal[:, 1:])
    m_time_50_kev = get_median(d_bin_edges[1:], h_time_50_kev[:, 1:])
    m_time_variable = get_median(d_bin_edges[1:], h_time_variable[:, 1:])
    m_energy_50_kev = get_median(d_bin_edges[1:], h_energy_50_kev[:, 1:])

    ii_order = np.arange(energy_thresholds.size)
    ii_order = np.flip(ii_order)

    # for ii in range(energy_thresholds.size):
    for ii in ii_order:
        ax00.stairs(h_ideal[ii, 1:] / np.sum(h_ideal[ii, 1:]), edges=d_bin_edges[1:], color=c_map(ii))
        # ax00.plot([m_ideal[ii], m_ideal[ii]], [0, 0.06], linestyle=':', color=c_map(ii))

        ax01.stairs(h_time_50_kev[ii, 1:] / np.sum(h_time_50_kev[ii, 1:]), edges=d_bin_edges[1:], color=c_map(ii))
        # ax01.plot([m_time_50_kev[ii], m_time_50_kev[ii]], [0, 0.06], linestyle=':', color=c_map(ii))

        ax10.stairs(h_time_variable[ii, 1:] / np.sum(h_time_variable[ii, 1:]), edges=d_bin_edges[1:], color=c_map(ii))
        # ax10.plot([m_time_variable[ii], m_time_variable[ii]], [0, 0.06], linestyle=':', color=c_map(ii))

        ax11.stairs(h_energy_50_kev[ii, 1:] / np.sum(h_energy_50_kev[ii, 1:]), edges=d_bin_edges[1:], color=c_map(ii))
        # ax11.plot([m_energy_50_kev[ii], m_energy_50_kev[ii]], [0, 0.06], linestyle=':', color=c_map(ii))

    d_energy_thresholds = energy_thresholds[1] - energy_thresholds[0]
    energy_thresholds_bounds = np.append(energy_thresholds - d_energy_thresholds / 2, energy_thresholds[-1] + d_energy_thresholds / 2) * 1e3
    sm = ScalarMappable(cmap=c_map, norm=BoundaryNorm(energy_thresholds_bounds, energy_thresholds.size))
    # sm.set_array([])
    cbar0 = fig.colorbar(sm, cax=axc0, orientation='vertical', ticks=[50, 150, 250, 350])
    # cbar0.set_label('Energy threshold [keV]')
    cbar0.set_label('Energy thresh. [keV]')
    cbar1 = fig.colorbar(sm, cax=axc1, orientation='vertical', ticks=[50, 150, 250, 350])
    # cbar1.set_label('Energy threshold [keV]')
    cbar1.set_label('Energy thresh. [keV]')

    ax00.set_xscale('log')
    ax00.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
    ax00.set_xticklabels([])
    # ax00.set_ylim(0, 0.06)
    # ax00.set_yticks([0, 0.02, 0.04, 0.06])
    ax00.set_ylim(0, 0.15)
    # ax00.set_yticks([0, 0.02, 0.04, 0.06])
    ax00.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax00.set_ylabel('Relative frequency')
    ax00.set_title('Ideal')

    ax01.set_xscale('log')
    ax01.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
    ax01.set_xticklabels([])
    # ax01.set_ylim(0, 0.06)
    # ax01.set_yticks([0, 0.02, 0.04, 0.06])
    ax01.set_ylim(0, 0.15)
    ax01.set_yticklabels([])
    ax01.set_title('Time-based\nat 50 keV')

    ax10.set_xscale('log')
    ax10.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
    # ax10.set_ylim(0, 0.06)
    # ax10.set_yticks([0, 0.02, 0.04, 0.06])
    ax10.set_ylim(0, 0.15)
    ax10.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax10.set_xlabel(r'$d_\mathrm{min}$ [mm]')
    ax10.set_ylabel('Relative frequency')
    ax10.set_title('Time-based\nvariable')

    ax11.set_xscale('log')
    ax11.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
    # ax11.set_ylim(0, 0.06)
    # ax11.set_yticks([0, 0.02, 0.04, 0.06])
    ax11.set_ylim(0, 0.15)
    ax11.set_yticklabels([])
    ax11.set_xlabel(r'$d_\mathrm{min}$ [mm]')
    ax11.set_title('Energy-based')

    plt.show()

    # ax1.set_ylim(0, 0.15)
    # ax1.set_yticks([0, 0.05, 0.10, 0.15])

    ax01.set_yticklabels(['', '', '', ''])
    ax01.set_xlabel(r'$d_\mathrm{min}$ [mm]')
    ax01.set_title('Variable energy selection')
    # ax1.set_ylabel(r'Relative frequency')
    ax01.set_title('variable')

    d_energy_thresholds = energy_thresholds[1] - energy_thresholds[0]
    energy_thresholds_bounds = np.append(energy_thresholds.flatten() - d_energy_thresholds / 2,
                                         energy_thresholds.flatten()[-1] + d_energy_thresholds / 2) * 1e3
    sm = ScalarMappable(cmap=c_map, norm=BoundaryNorm(energy_thresholds_bounds, energy_thresholds.size))
    sm.set_array([])

    # cbar = fig.colorbar(sm, cax=axc, orientation='vertical', ticks=[50, 150, 250, 350])
    # cbar.set_label('Energy threshold [keV]')

    fig.suptitle('Energy selection:', fontweight='bold')

    plt.show()

    # sys.exit()

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    ax.plot(energy_thresholds * 1e3, median_d_min, color='tab:blue', linestyle='--')
    ax.plot(energy_thresholds * 1e3, median_d_min_v2, color='tab:blue', linestyle='-')
    ax_twin.plot(energy_thresholds * 1e3, h_0 * 100, color='tab:orange', linestyle='--')
    ax_twin.plot(energy_thresholds * 1e3, h_0_v2 * 100, color='tab:orange', linestyle='-')

    ax.set_xlabel('Energy threshold [keV]')
    ax.set_ylabel(r'Median [mm] of $d_\mathrm{min}\geq10^{-4}$ mm', color='tab:blue')
    ax.tick_params(axis='y', colors='tab:blue')
    ax.spines['left'].set_color('tab:blue')
    ax.set_ylim(15, 175)

    ax_twin.set_ylabel(r'Proportion [%] of $d_\mathrm{min}<10^{-4}$ mm', color='tab:orange')
    ax_twin.tick_params(axis='y', colors='tab:orange')
    ax_twin.spines['right'].set_color('tab:orange')
    # ax_twin.set_ylim(65-1, 90+1)
    ax_twin.set_ylim(0 - 1, 70 + 1)

    ax.plot(np.nan, color='k', linestyle='-', label='variable')
    ax.plot(np.nan, color='k', linestyle='--', label='at 50 keV')
    # ax.legend(loc='lower left', frameon=False, title='Event selection:', title_fontproperties={'weight': 'bold'})
    ax.legend(loc='upper center', frameon=False, title='Event selection:', title_fontproperties={'weight': 'bold'})
    plt.show()

    return 0


def get_median(d_bin_edges, hist, vis=False):
    d_bin_centers = (d_bin_edges[:-1] + d_bin_edges[1:]) / 2
    hist_norm = hist / hist.sum(axis=1)[:, np.newaxis]
    cummulative_mass_function = np.cumsum(hist_norm, axis=1)

    if vis:
        fig, ax = plt.subplots()
        ax.plot(d_bin_centers, cummulative_mass_function.T)
        ax.set_xscale('log')
        plt.show()

    # mean_d_min = np.sum(hist_norm * d_bin_centers[np.newaxis, :], axis=1)
    median_d_min = np.apply_along_axis(lambda var: np.interp(0.5, var, d_bin_centers), 1, cummulative_mass_function)

    # return mean_d_min
    return median_d_min


if __name__ == "__main__":
    # main()
    plot_d_min_distributions()
