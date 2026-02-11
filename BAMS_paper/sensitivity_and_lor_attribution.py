"""
Sensitivity and the frequency of incorrect LOR attribution as a function of the energy threshold

@author: Martin Rädler
"""
# Python libraries
import sys
from time import sleep
from glob import glob
from natsort import natsorted
from uproot import open as open_root
from scipy.constants import speed_of_light
from tqdm import trange, tqdm
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array
from CASToR.lut import read_lut_binary, read_lut_header
from CASToR.root_to_cdf import extract_coincidence_data, visualize_crystal_attribution, visualize_lors, analyze_deviation
from CASToR.npy_to_cdf import get_castor_id2


def main():
    lut = read_lut_binary(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert.lut')
    lut_header = read_lut_header(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert.hscan')

    # root_files = natsorted(glob('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-04-15_10-32-22/results_*.root'))  # Line source
    root_files = natsorted(glob('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-05-21_09-14-48/results_*.root'))  # Extended source
    # root_files = natsorted(glob('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-05-22_09-58-47/results_*.root'))  # Water phantom added
    # root_files = natsorted(glob('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-05-26_12-36-43/results_*.root'))  # Threshold set to zero
    root_files = root_files[0:1]
    [print(entry) for entry in root_files]

    # Keys necessary to calculate the sensitivity map
    necessary_keys = ['eventID1', 'sourceID1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'time1', 'energy1', 'globalPosX1', 'globalPosY1', 'globalPosZ1', 'gantryID1', 'rsectorID1', 'moduleID1', 'submoduleID1', 'crystalID1', 'layerID1', 'comptonCrystal1', 'RayleighCrystal1', 'comptonPhantom1', 'RayleighPhantom1',
                      'eventID2', 'sourceID2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'time2', 'energy2', 'globalPosX2', 'globalPosY2', 'globalPosZ2', 'gantryID2', 'rsectorID2', 'moduleID2', 'submoduleID2', 'crystalID2', 'layerID2', 'comptonCrystal2', 'RayleighCrystal2', 'comptonPhantom2', 'RayleighPhantom2']

    for ii in trange(len(root_files)):
        # Load the simulation data
        root_file = open_root(root_files[ii])
        # for key, data_type in root_file['MergedCoincidences'].itertypenames():
        #     print(key)
        coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'], keys=necessary_keys, overwrite=False)

    # Use only 10 %
    coincidences_struct = coincidences_struct[:int(coincidences_struct.size * .1)]

    not_phantom_scattered = filter_phantom_scattered(coincidences_struct)

    true = filter_true(coincidences_struct, verbose=False)
    true = true & not_phantom_scattered

    (x1, y1, z1, t1, gantry_id1, rsector_id1, crystal_id1, layer_id1,
     x2, y2, z2, t2, gantry_id2, rsector_id2, crystal_id2, layer_id2, num_entries) = extract_coincidence_data(coincidences_struct)

    c1 = get_castor_id2(gantry_id1, rsector_id1, crystal_id1, layer_id1)
    c2 = get_castor_id2(gantry_id2, rsector_id2, crystal_id2, layer_id2)

    # layer_id_grid = np.arange(16 * 110)
    # layer_id_test = np.array(512, ndmin=1)
    #
    # c_grid = get_castor_id2(np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), layer_id_grid)
    # c_test = get_castor_id2(np.zeros(layer_id_test.size, dtype=int), np.zeros(layer_id_test.size, dtype=int), np.zeros(layer_id_test.size, dtype=int), layer_id_test)
    #
    # _,zz = np.unravel_index(layer_id_test, (16, 110), order='F')
    # print(zz)
    #
    # fig, ax = plt.subplots()
    # ax.scatter(lut[c_grid]['Posy'], lut[c_grid]['Posz'], c=layer_id_grid)
    # ax.plot(lut[c_test]['Posy'], lut[c_test]['Posz'], 'x', color='tab:red')
    # plt.show()

    dt_0 = blur_time(t2 - t1, dt_fwhm_ps=0.)
    dt_200 = blur_time(t2 - t1, dt_fwhm_ps=200.)
    dt_400 = blur_time(t2 - t1, dt_fwhm_ps=400.)
    dt_600 = blur_time(t2 - t1, dt_fwhm_ps=600.)

    # subset = np.arange(num_entries)
    # subset = np.random.choice(num_entries, replace=False, size=min(1000000, num_entries))
    # visualize_crystal_attribution(x1[subset], y1[subset], z1[subset], lut[c1[subset]])
    # visualize_crystal_attribution(x2[subset], y2[subset], z2[subset], lut[c2[subset]])
    # visualize_lors(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], lut[c1[subset]], lut[c2[subset]])
    # analyze_deviation(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], c1[subset], c2[subset], lut, lut_header)
    # verify_tof_sign(sx[subset], sy[subset], sz[subset], lut[c1[subset]], lut[c2[subset]], delta_t[subset])

    # Scatter test
    scatter_test_plot = False
    if scatter_test_plot:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(8, 5))
        pass_adjacency_test = run_adjacency_test(gantry_id1, rsector_id1, gantry_id2, rsector_id2, minimum_sector_difference=2, vis=False)
        # pass_scatter_test = run_scatter_test(gantry_id1, gantry_id2, c1, c2, dt, true, lut, vis_ax=True)
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_0[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax)
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_200[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax, linestyle='-.')
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_400[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax, linestyle='--')
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_600[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax, linestyle=':')

        sys.exit()

        ax_sub = fig.add_axes([0.55, 0.475, 0.34, 0.35])
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_0[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax_sub)
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_200[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax_sub, linestyle='-.')
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_400[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax_sub, linestyle='--')
        pass_scatter_test = run_scatter_test(gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],
                                             c1[pass_adjacency_test], c2[pass_adjacency_test], dt_600[pass_adjacency_test],
                                             true[pass_adjacency_test], lut, vis_ax=ax_sub, linestyle=':')

        # pass_scatter_test = run_scatter_test(c1[pass_adjacency_test], c2[pass_adjacency_test], dt[pass_adjacency_test], lut, vis_ax=False)
        ax.set_xlim(-100, 300)
        # ax.set_ylim(0, 0.014)
        ax.set_ylim(0, 0.01)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel(r'$|\mathbf{r}_2-\mathbf{r}_1|-(t_2-t_1)/c_0$ [cm]')
        ax.set_ylabel('Normalized distribution [1/cm]')

        p0, = ax.plot(np.nan, color='k', linestyle='-')
        p1, = ax.plot(np.nan, color='k', linestyle='-.')
        p2, = ax.plot(np.nan, color='k', linestyle='--')
        p3, = ax.plot(np.nan, color='k', linestyle=':')
        legend = ax.legend(handles=[p0, p1, p2, p3], labels=['0', '200', '400', '600'],
                           loc='upper left', frameon=False, title=r'$\bf{FWHM}$' '\n' r'$\mathbf{\Delta t\,[ps]}$')

        # p0, = ax.plot(np.nan, color='k')
        p1, = ax.plot(np.nan, color='tab:blue')
        p2, = ax.plot(np.nan, color='tab:orange')
        p3, = ax.plot(np.nan, color='tab:green')
        p4, = ax.plot(np.nan, color='tab:red')
        ax.legend(handles=[p1, p2, p3, p4], labels=['Comb.', 'TB-TB', 'TB-BI', 'BI-BI'],
                  loc='lower right', frameon=False, title=r'$\bf{True\,\,coincidences}$', ncol=2)
        # ax.legend(handles=[p1, p2, p3, p4], labels=['Comb.', 'TB-TB', 'TB-BI', 'BI-BI'],
        #           loc='lower right', frameon=False, title=r'$\bf{True\,\,coinc.\,(in\,\,det.)}$', ncol=2)

        ax.add_artist(legend)

        ax.annotate(r'$\bf{All}$' '\n' r'$\bf{coinc.}$', xy=(70, 1e-2), xytext=(10, 1e-2), ha='left', va='center',
                     arrowprops=dict(arrowstyle='->', color='k'))

        ax_sub.set_xlim(-5, 55)
        # ax_sub.set_ylim(0, 6e-4)
        # ax_sub.set_ylim(0, 8e-4)
        ax_sub.set_ylim(0, 2.4e-4)
        ax_sub.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax_sub.set_yscale('log')
        # ax_sub.set_ylim(1e-6, 1e-1)
        ax_sub.set_xticks([0, 25, 50])
        # ax_sub.set_yticks([0, 2e-4, 4e-4, 6e-4, 8e-4])
        ax_sub.set_yticks([0, 1e-4, 2e-4])
        plt.show()

    pass_adjacency_test = run_adjacency_test(gantry_id1, rsector_id1, gantry_id2, rsector_id2, minimum_sector_difference=2, vis=False)
    pass_scatter_test = run_scatter_test(gantry_id1, gantry_id2, c1, c2, dt_0, true, lut, vis_ax=False)
    coincidences_struct_filtered = coincidences_struct[pass_adjacency_test & pass_scatter_test]
    not_phantom_scattered = filter_phantom_scattered(coincidences_struct_filtered)

    # fig, ax = plt.subplots()
    # multiplicity, window_sizes, group_sizes, group_indices = get_multiplicity(coincidences_struct)
    # multiplicity_analysis(multiplicity, window_sizes, group_sizes, group_indices, vis_ax=ax)
    multiplicity, window_sizes, group_sizes, group_indices = get_multiplicity(coincidences_struct_filtered)
    multiplicity_analysis(multiplicity, window_sizes, group_sizes, group_indices, vis_ax=True)
    # plt.show()

    # check_event_ordering(coincidences_struct_filtered, multiplicity, group_indices, group_sizes)
    coincidences_struct_filtered_2 = blur_time_and_reorder(coincidences_struct_filtered, multiplicity, group_indices, group_sizes, fwhm_ps=500)

    true_filtered = filter_true(coincidences_struct_filtered, verbose=False)
    true_filtered_2 = filter_true(coincidences_struct_filtered_2, verbose=False)

    e_sel = energy_based_selection(coincidences_struct_filtered, group_indices)
    ff_3 = np.sum(true_filtered[e_sel]) / e_sel.size * 100

    print(ff_3)
    # sys.exit()


    ff = np.sum(true_filtered[group_indices[:-1]]) / (group_indices.size - 1) * 100
    ff_2 = np.sum(true_filtered_2[group_indices[:-1]]) / (group_indices.size - 1) * 100

    print(ff)
    print(ff_2)
    fwhm = [0, 100, 200, 300, 400, 500, 600]
    hh = [81.63917164949514, 77.26064181786631, 73.6165951232855, 71.02067766750146, 69.07012585183072, 67.51758802837614, 66.3673048020447]

    fig, ax = plt.subplots()
    ax.plot(fwhm, hh)
    plt.show()

    # sys.exit()


    #
    energy_thresholds = np.linspace(50, 350, 20)[:, np.newaxis] * 1e-3  # MeV

    event_selection = event_selection_multiplicity_energy(multiplicity, group_sizes, group_indices,
                                                          coincidences_struct['energy1'][pass_adjacency_test & pass_scatter_test],
                                                          coincidences_struct['energy2'][pass_adjacency_test & pass_scatter_test],
                                                          energy_thresholds)

    # filter_true(coincidences_struct, verbose=False)


    # multiplicity_selection(multiplicity, window_sizes, group_sizes, group_indices, coincidences_struct_filtered, true_filtered)

    sensitivity_rise_plot = True
    if sensitivity_rise_plot:
        n_events, p_true, p_phantom_scatter = energy_threshold_variation(energy_thresholds, multiplicity, window_sizes, group_sizes, group_indices, coincidences_struct_filtered, true_filtered, not_phantom_scattered)
        n_events_v2, p_true_v2, p_phantom_scatter_v2 = energy_threshold_variation_v2(energy_thresholds, event_selection, coincidences_struct_filtered, true_filtered, not_phantom_scattered)

        reference = np.interp(200e-3, energy_thresholds.flatten(), n_events_v2.flatten())

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(energy_thresholds * 1e3, n_events / reference, color='tab:blue', linestyle='--')
        ax.plot(energy_thresholds * 1e3, n_events_v2 / reference, color='tab:blue', linestyle='-')
        ax.set_xlabel('Energy threshold $E$ [keV]')
        ax.tick_params(axis='y', colors='tab:blue')
        ax.spines['left'].set_color('tab:blue')
        ax.set_ylabel(r'$N_c(E)/N_c$(200 keV)', color='tab:blue')

        ax.plot(np.nan, color='k', linestyle='-', label='variable')
        ax.plot(np.nan, color='k', linestyle='--', label='at 50 keV')
        ax.legend(loc='center', frameon=False, title='Event selection:', title_fontproperties={'weight': 'bold'})

        ax_twin = ax.twinx()
        ax_twin.plot(energy_thresholds * 1e3, p_true, color='tab:orange', linestyle='--')
        ax_twin.plot(energy_thresholds * 1e3, p_true_v2, color='tab:orange', linestyle='-')

        ax_twin.plot(energy_thresholds * 1e3, p_phantom_scatter, color='tab:orange', linestyle='--')
        ax_twin.plot(energy_thresholds * 1e3, p_phantom_scatter_v2, color='tab:orange', linestyle='-')

        ax_twin.set_ylim(0, 1)
        ax_twin.tick_params(axis='y', colors='tab:orange')
        ax_twin.spines['right'].set_color('tab:orange')
        ax_twin.set_ylabel('Proportion [%]', color='tab:orange')

        ax_twin.annotate('True\n in detector', xy=(300, 0.88), xytext=(300, 0.75), color='tab:orange', ha='center', va='center',
                         arrowprops=dict(arrowstyle='->', color='tab:orange'))
        ax_twin.annotate('True\n in phantom', xy=(300, 0.35), xytext=(300, 0.22), color='tab:orange', ha='center', va='center',
                         arrowprops=dict(arrowstyle='->', color='tab:orange'))

        plt.show()

    # multiplicity plots?
    # with phantom
    # median of the scatterd ones
    #

    median_d_min_plot = False
    if median_d_min_plot:

        plt.rcParams.update({'font.size': 16})
        fig, (ax0, ax1, axc) = plt.subplots(1, 3, figsize=(8, 4), width_ratios=(1, 1, 0.05))
        c_map = get_cmap('viridis', energy_thresholds.size)

        median_d_min, h_0 = lor_source_point_distance_distribution(coincidences_struct_filtered, energy_thresholds, group_indices, ax0, c_map)
        median_d_min_v2, h_0_v2 = lor_source_point_distance_distribution_v2(coincidences_struct_filtered, energy_thresholds, event_selection, ax1, c_map)

        ax0.set_xscale('log')
        ax0.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
        ax0.set_ylim(0, 0.06)
        ax0.set_yticks([0, 0.02, 0.04, 0.06])

        # ax0.set_ylim(0, 0.15)
        # ax0.set_yticks([0, 0.05, 0.10, 0.15])

        ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax0.set_xlabel(r'$d_\mathrm{min}$ [mm]')
        ax0.set_ylabel(r'Relative frequency')
        ax0.set_title('at 50 keV')

        ax1.set_xscale('log')
        ax1.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
        ax1.set_ylim(0, 0.06)
        ax1.set_yticks([0, 0.02, 0.04, 0.06])

        # ax1.set_ylim(0, 0.15)
        # ax1.set_yticks([0, 0.05, 0.10, 0.15])

        ax1.set_yticklabels(['', '', '', ''])
        ax1.set_xlabel(r'$d_\mathrm{min}$ [mm]')
        ax1.set_title('Variable energy selection')
        # ax1.set_ylabel(r'Relative frequency')
        ax1.set_title('variable')

        d_energy_thresholds = energy_thresholds[1] - energy_thresholds[0]
        energy_thresholds_bounds = np.append(energy_thresholds.flatten() - d_energy_thresholds / 2, energy_thresholds.flatten()[-1] + d_energy_thresholds / 2) * 1e3
        sm = ScalarMappable(cmap=c_map, norm=BoundaryNorm(energy_thresholds_bounds, energy_thresholds.size))
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=axc, orientation='vertical', ticks=[50, 150, 250, 350])
        cbar.set_label('Energy threshold [keV]')

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
        ax_twin.set_ylim(0-1, 70+1)

        ax.plot(np.nan, color='k', linestyle='-', label='variable')
        ax.plot(np.nan, color='k', linestyle='--', label='at 50 keV')
        # ax.legend(loc='lower left', frameon=False, title='Event selection:', title_fontproperties={'weight': 'bold'})
        ax.legend(loc='upper center', frameon=False, title='Event selection:', title_fontproperties={'weight': 'bold'})
        plt.show()

    return 0


def run_scatter_test(gantry_id1, gantry_id2, c1, c2, dt, true, lut, vis_ax=False, linestyle='-'):

    tbtb = (gantry_id1 < 2) & (gantry_id2 < 2)
    tbbi = ((gantry_id1 < 2) & (gantry_id2 == 2)) | (gantry_id1 == 2) & (gantry_id2 < 2)
    bibi = (gantry_id1 == 2) & (gantry_id2 == 2)

    lut1, lut2 = lut[c1], lut[c2]
    dr = np.sqrt((lut1['Posx'] - lut2['Posx']) ** 2 + (lut1['Posy'] - lut2['Posy']) ** 2 + (lut1['Posz'] - lut2['Posz']) ** 2) * 1e-3  # [m]
    scatter_test = (dr - dt * speed_of_light) * 100 # cm

    bin_edges = np.linspace(-300, 300, 500 + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1:] - bin_edges[:-1]

    h, _ = np.histogram(scatter_test, bins=bin_edges)
    h_true, _ = np.histogram(scatter_test[true], bins=bin_edges)
    h_tbtb, _ = np.histogram(scatter_test[true & tbtb], bins=bin_edges)
    h_tbbi, _ = np.histogram(scatter_test[true & tbbi], bins=bin_edges)
    h_bibi, _ = np.histogram(scatter_test[true & bibi], bins=bin_edges)

    # norm = np.sum(h)
    norm = np.trapz(h, x=bin_centers)
    h, h_true, h_tbtb, h_tbbi, h_bibi = h.astype(float) / norm, h_true.astype(float) / norm, h_tbtb.astype(float) / norm, h_tbbi.astype(float) / norm, h_bibi.astype(float) / norm

    # Determine the minimum between the zero peak and the peak attributable to the shortest LOR with a low time
    # difference, i.e. originating from the center (about 30 cm)
    # search_interval = (bin_centers >= 0) & (bin_centers <= 0.3)
    search_interval = (bin_centers >= 0) & (bin_centers <= 0.3 * 100)
    idx_min = np.argmin(h[search_interval])
    threshold = bin_centers[search_interval][idx_min]
    print(threshold)
    if threshold > 10.:
        threshold = 10.
        print('Threshold ')

    print(np.sum(h_bibi[bin_centers >= threshold]) / np.sum(h_bibi))

    if vis_ax:
        is_axes = isinstance(vis_ax, Axes)
        if not is_axes:
            fig, vis_ax = plt.subplots()
        vis_ax.stairs(h, edges=bin_edges, color='black', linestyle=linestyle)
        vis_ax.stairs(h_true, edges=bin_edges, color='tab:blue', linestyle=linestyle)
        vis_ax.stairs(h_tbtb, edges=bin_edges, color='tab:orange', linestyle=linestyle)
        vis_ax.stairs(h_tbbi, edges=bin_edges, color='tab:green', linestyle=linestyle)
        vis_ax.stairs(h_bibi, edges=bin_edges, color='tab:red', linestyle=linestyle)
        vis_ax.plot([threshold, threshold], [0, h[search_interval][idx_min]], color='k', linestyle=linestyle)
        if not is_axes:
            plt.show()

    # pass_scatter_test = np.ones(t1.size, dtype=bool)
    # pass_scatter_test[scatter_test < threshold] = False
    pass_scatter_test = scatter_test > threshold

    print('Passing scatter test: %1.2f %%.' % (np.sum(pass_scatter_test) / pass_scatter_test.size * 100))

    return pass_scatter_test


def blur_time(t, dt_fwhm_ps=500.):
    """
    :param t: Array of interaction times in units of seconds
    :param dt_fwhm_ps: Time resolution in picoseconds as FWHM
    :return: Blurred time array
    """
    dt_sigma = dt_fwhm_ps * 1e-12 / (2 * np.sqrt(2 * np.log(2)))  # [s]
    t += np.random.normal(loc=0., scale=dt_sigma, size=t.size)
    return t


def run_adjacency_test(gantry_id1, rsector_id1, gantry_id2, rsector_id2, minimum_sector_difference=2, vis=False):
    # Treat the TB-J-PET as a single scanner
    new_gantry_id1 = gantry_id1.copy()
    new_gantry_id1[new_gantry_id1 == 1] = 0
    new_gantry_id1[new_gantry_id1 == 2] = 1

    new_gantry_id2 = gantry_id2.copy()
    new_gantry_id2[new_gantry_id2 == 1] = 0
    new_gantry_id2[new_gantry_id2 == 2] = 1

    ravel_id1 = np.ravel_multi_index((new_gantry_id1, rsector_id1), (2, 24), order='C')
    ravel_id2 = np.ravel_multi_index((new_gantry_id2, rsector_id2), (2, 24), order='C')

    id_diff = np.abs(ravel_id1 - ravel_id2)

    if vis:
        h = np.bincount(id_diff)
        fig, ax = plt.subplots()
        ax.bar(np.arange(h.size), h, width=0.8)
        plt.show()

    pass_adjacency_test = id_diff >= minimum_sector_difference

    print('Passing adjacency test: %1.2f %%.' % (np.sum(pass_adjacency_test) / pass_adjacency_test.size * 100))

    return pass_adjacency_test


def filter_true(coincidences_struct, verbose=False):
    event_id_1, event_id_2 = coincidences_struct['eventID1'], coincidences_struct['eventID2']
    compton_crystal_1, compton_crystal_2 = coincidences_struct['comptonCrystal1'], coincidences_struct['comptonCrystal2']
    rayleigh_crystal_1, rayleigh_crystal_2 = coincidences_struct['RayleighCrystal1'], coincidences_struct['RayleighCrystal2']

    event_id_check = event_id_1 == event_id_2
    compton_scatter_check = (compton_crystal_1 == 1) & (compton_crystal_2 == 1)
    rayleigh_scatter_check = (rayleigh_crystal_1 == 0) & (rayleigh_crystal_2 == 0)

    true = event_id_check & compton_scatter_check & rayleigh_scatter_check

    if verbose:
        print('Passing eventID check: %1.2f %%.' % (np.sum(event_id_check) / event_id_check.size * 100))
        print('Passing Compton scatter check: %1.2f %%.' % (np.sum(compton_scatter_check) / compton_scatter_check.size * 100))
        print('Passing Rayleigh scatter check: %1.2f %%.' % (np.sum(rayleigh_scatter_check) / rayleigh_scatter_check.size * 100))
        print('Overall: %1.2f %%' % (np.sum(true) / true.size * 100))
    return true


def filter_phantom_scattered(coincidences_struct, vis=False):

    compton_phantom1, rayleigh_phantom1 = coincidences_struct['comptonPhantom1'], coincidences_struct['RayleighPhantom1']
    compton_phantom2, rayleigh_phantom2 = coincidences_struct['comptonPhantom2'], coincidences_struct['RayleighPhantom2']
    not_phantom_scattered = (compton_phantom1 == 0) & (rayleigh_phantom1 == 0) & (compton_phantom2 == 0) & (rayleigh_phantom2 == 0)

    print('%1.2f %%' % (np.sum(not_phantom_scattered) / not_phantom_scattered.size * 100))

    if vis:
        compton_phantom1_stats, rayleigh_phantom1_stats = np.bincount(compton_phantom1), np.bincount(rayleigh_phantom1)
        compton_phantom2_stats, rayleigh_phantom2_stats = np.bincount(compton_phantom2), np.bincount(rayleigh_phantom2)

        fig, ax = plt.subplots()
        ax.bar(np.arange(compton_phantom1_stats.size), compton_phantom1_stats, alpha=0.75)
        ax.bar(np.arange(compton_phantom2_stats.size), compton_phantom2_stats, alpha=0.75)
        # ax.bar(np.arange(rayleigh_phantom1_stats.size), rayleigh_phantom1_stats, alpha=0.75)
        # ax.bar(np.arange(rayleigh_phantom2_stats.size), rayleigh_phantom2_stats, alpha=0.75)
        ax.set_yscale('log')
        plt.show()
        sys.exit()

    return not_phantom_scattered


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


def multiplicity_analysis(multiplicity, window_sizes, group_sizes, group_indices, vis_ax=False):
    multiplicity_histogram = np.bincount(multiplicity)

    bin_edges = np.linspace(0, 10e-9, 100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    window_size_histogram, _ = np.histogram(window_sizes, bins=bin_edges)

    if vis_ax:
        is_axes = isinstance(vis_ax, Axes)
        if not is_axes:
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
        if not is_axes:
            plt.show()
    return 0


def multiplicity_selection(multiplicity, window_sizes, group_sizes, group_indices, coincidences_struct, true):
    time_1, time_2 = coincidences_struct['time1'], coincidences_struct['time2']

    accuracy_mult_2 = []
    accuracy_mult_3 = []
    accuracy_mult_4 = []
    accuracy_mult_5 = []

    mult_2_except = 0

    for ii in trange(multiplicity.size):
        group = np.arange(group_indices[ii], group_indices[ii + 1])
        # print(group.size - group_sizes[ii])
        if multiplicity[ii] == 2:
            if group.size > 1:
                mult_2_except += 1
            # Choosing the first element is only necessary, since some of the events were miss-classified, based on time
            accuracy_mult_2.append(true[group][0])
        if multiplicity[ii] == 3:
            # print(true[group])
            tt = np.mean(np.vstack((time_1[group], time_2[group])), axis=0)
            if np.any(tt[:-1] > tt[1:]):
                # print('Exception')
                # print(tt)
                pass

            accuracy_mult_3.append(true[group][0])

            # A random event rarely successfully opens a window

        if multiplicity[ii] == 4:
            accuracy_mult_4.append(true[group][0])

        if multiplicity[ii] == 5:
            accuracy_mult_5.append(true[group][0])

    print(mult_2_except)

    accuracy_mult_2 = np.array(accuracy_mult_2)
    accuracy_mult_3 = np.array(accuracy_mult_3)
    accuracy_mult_4 = np.array(accuracy_mult_4)
    accuracy_mult_5 = np.array(accuracy_mult_5)

    print(np.sum(accuracy_mult_2) / accuracy_mult_2.size * 100)
    print(np.sum(accuracy_mult_3) / accuracy_mult_3.size * 100)
    print(np.sum(accuracy_mult_4) / accuracy_mult_4.size * 100)
    print(np.sum(accuracy_mult_5) / accuracy_mult_5.size * 100)

    return 0


def energy_threshold_variation(energy_thresholds, multiplicity, window_sizes, group_sizes, group_indices, coincidences_struct, true, not_phantom_scattered):
    energy_1, energy_2 = coincidences_struct['energy1'], coincidences_struct['energy2']

    group_first_indices = group_indices[:-1]
    print(np.sum(true[group_first_indices]) / group_first_indices.size)

    multiplicity_element_wise = np.repeat(multiplicity, group_sizes)

    for ii in range(2, np.max(multiplicity) + 1):
        p_true_first = np.sum(true[group_first_indices[multiplicity == ii]]) / np.sum(multiplicity == ii) * 100
        p_true = np.sum(true[multiplicity_element_wise == ii]) / np.sum(multiplicity == ii) * 100

        print('%d: %1.2f < %1.2f %% (%1.2f)' % (ii, p_true_first, p_true, p_true_first / p_true * 100))

    # sys.exit()

    # m2_e1 = energy_1[multiplicity_element_wise == 2][np.newaxis, :]
    # m2_e2 = energy_2[multiplicity_element_wise == 2][np.newaxis, :]
    # m2_t = true[multiplicity_element_wise == 2][np.newaxis, :]

    # m2_e1 = energy_1[group_first_indices[multiplicity == 5]][np.newaxis, :]
    # m2_e2 = energy_2[group_first_indices[multiplicity == 5]][np.newaxis, :]
    # m2_t = true[group_first_indices[multiplicity == 5]][np.newaxis, :]

    m2_e1 = energy_1[group_first_indices][np.newaxis, :]
    m2_e2 = energy_2[group_first_indices][np.newaxis, :]
    m2_t = true[group_first_indices][np.newaxis, :]
    m2_nps = not_phantom_scattered[group_first_indices][np.newaxis, :]

    #
    xx = (m2_e1 >= energy_thresholds) & (m2_e2 >= energy_thresholds)
    n_events = np.sum(xx, axis=1)
    p_true = np.sum(xx & m2_t, axis=1) / n_events
    p_phantom_scatter = np.sum(xx & m2_nps, axis=1) / n_events

    return n_events, p_true, p_phantom_scatter


def lor_source_point_distance_distribution(coincidences_struct, energy_thresholds, group_indices, ax, c_map):
    group_first_indices = group_indices[:-1]
    group_first_boolean = np.zeros(coincidences_struct.size, dtype=bool)
    group_first_boolean[group_first_indices] = True

    same_event_ids = coincidences_struct['eventID1'] == coincidences_struct['eventID2']

    compton_phantom1, rayleigh_phantom1 = coincidences_struct['comptonPhantom1'], coincidences_struct['RayleighPhantom1']
    compton_phantom2, rayleigh_phantom2 = coincidences_struct['comptonPhantom2'], coincidences_struct['RayleighPhantom2']
    not_phantom_scattered = (compton_phantom1 == 0) & (rayleigh_phantom1 == 0) & (compton_phantom2 == 0) & (rayleigh_phantom2 == 0)
    same_event_ids = same_event_ids & (~ not_phantom_scattered)

    source_pos = np.stack((coincidences_struct['sourcePosX1'],
                           coincidences_struct['sourcePosY1'],
                           coincidences_struct['sourcePosZ1']), axis=1)

    global_pos_1 = np.stack((coincidences_struct['globalPosX1'],
                             coincidences_struct['globalPosY1'],
                             coincidences_struct['globalPosZ1']), axis=1)

    global_pos_2 = np.stack((coincidences_struct['globalPosX2'],
                             coincidences_struct['globalPosY2'],
                             coincidences_struct['globalPosZ2']), axis=1)

    source_pos = source_pos[group_first_boolean & same_event_ids, :]
    global_pos_1 = global_pos_1[group_first_boolean & same_event_ids, :]
    global_pos_2 = global_pos_2[group_first_boolean & same_event_ids, :]

    d_min = (np.linalg.norm(np.cross(source_pos - global_pos_1, global_pos_2 - global_pos_1, axis=1), axis=1) /
             np.linalg.norm(global_pos_2 - global_pos_1, axis=1))

    energy_1, energy_2 = coincidences_struct['energy1'], coincidences_struct['energy2']
    energy_1 = energy_1[group_first_boolean & same_event_ids]
    energy_2 = energy_2[group_first_boolean & same_event_ids]

    # d_bin_edges = np.insert(np.logspace(-6, 4, 100), 0, 0)
    d_bin_edges = np.logspace(-4, 4, 100)
    # d_bin_edges = np.insert(d_bin_edges, 0, 0)
    d_bin_centers = (d_bin_edges[:-1] + d_bin_edges[1:]) / 2
    d_bin_widths = d_bin_edges[1:] - d_bin_edges[:-1]

    d_min_idx = np.digitize(d_min, bins=d_bin_edges) - 1

    # energy_thresholds = np.linspace(50, 350, 20)[:, np.newaxis] * 1e-3  # MeV

    h_2d = np.zeros((energy_thresholds.size, d_bin_centers.size))
    h_0 = np.zeros(energy_thresholds.size)

    for ii in range(energy_thresholds.size):
        above_energy_threshold = (energy_1 >= energy_thresholds[ii]) & (energy_2 >= energy_thresholds[ii])
        d_min_idx_above_energy_threshold = d_min_idx[above_energy_threshold]
        h_2d[ii, :] = np.bincount(d_min_idx_above_energy_threshold[d_min_idx_above_energy_threshold >= 0], minlength=d_bin_centers.size)
        h_0[ii] = np.sum(d_min_idx_above_energy_threshold < d_bin_edges[0]) / d_min_idx_above_energy_threshold.size

        # h_2d[ii, :] /= np.sum(h_2d[ii, :] * d_bin_widths)
        h_2d[ii, :] /= np.sum(h_2d[ii, :])

        # ax.stairs(h_2d[ii, :], edges=d_bin_edges, color=c_map(ii / (energy_thresholds.size - 1)))
        ax.stairs(h_2d[ii, :], edges=d_bin_edges, color=c_map(ii))

    mean_d_min = np.sum(h_2d * d_bin_centers[np.newaxis, :], axis=1)

    cmf = np.cumsum(h_2d, axis=1)
    median_d_min = np.apply_along_axis(lambda var: np.interp(0.5, var, d_bin_centers), 1, cmf)

    # fig, ax = plt.subplots()
    # ax.plot(energy_thresholds, mean_d_min)
    # ax.plot(energy_thresholds, median_d_min)
    # # ax.plot(d_bin_centers, cmf.T)
    # # ax.set_xscale('log')
    # plt.show()

    # h = np.bincount(d_min_idx[d_min_idx >= 0], minlength=d_bin_centers.size)
    # fig, ax = plt.subplots()
    # ax.bar(d_bin_centers, h, width=d_bin_widths)
    # ax.set_xscale('log')
    # plt.show()


    # print(np.sum(source_pos_x_1 == source_pos_x_2) / source_pos_x_1.size)
    # print(np.sum(source_pos_x_1[same_event_ids] == source_pos_x_2[same_event_ids]) / source_pos_x_1[same_event_ids].size)
    # print(coincidences_struct.dtype)

    # sys.exit()

    return median_d_min, h_0


def event_selection_multiplicity_energy(multiplicity, group_sizes, group_indices, energy_1, energy_2, energy_thresholds):

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


def lor_source_point_distance_distribution_v2(coincidences_struct, energy_thresholds, event_selection, ax, c_map):

    same_event_ids = coincidences_struct['eventID1'] == coincidences_struct['eventID2']
    compton_phantom1, rayleigh_phantom1 = coincidences_struct['comptonPhantom1'], coincidences_struct['RayleighPhantom1']
    compton_phantom2, rayleigh_phantom2 = coincidences_struct['comptonPhantom2'], coincidences_struct['RayleighPhantom2']
    not_phantom_scattered = (compton_phantom1 == 0) & (rayleigh_phantom1 == 0) & (compton_phantom2 == 0) & (rayleigh_phantom2 == 0)
    same_event_ids = same_event_ids & (~ not_phantom_scattered)

    source_pos = np.stack((coincidences_struct['sourcePosX1'],
                           coincidences_struct['sourcePosY1'],
                           coincidences_struct['sourcePosZ1']), axis=1)

    global_pos_1 = np.stack((coincidences_struct['globalPosX1'],
                             coincidences_struct['globalPosY1'],
                             coincidences_struct['globalPosZ1']), axis=1)

    global_pos_2 = np.stack((coincidences_struct['globalPosX2'],
                             coincidences_struct['globalPosY2'],
                             coincidences_struct['globalPosZ2']), axis=1)

    d_min = (np.linalg.norm(np.cross(source_pos - global_pos_1, global_pos_2 - global_pos_1, axis=1), axis=1) /
             np.linalg.norm(global_pos_2 - global_pos_1, axis=1))

    d_bin_edges = np.logspace(-4, 4, 100)
    # d_bin_edges = np.insert(d_bin_edges, 0, 0)
    d_bin_centers = (d_bin_edges[:-1] + d_bin_edges[1:]) / 2
    d_bin_widths = d_bin_edges[1:] - d_bin_edges[:-1]

    d_min_idx = np.digitize(d_min, bins=d_bin_edges) - 1

    h_2d = np.zeros((energy_thresholds.size, d_bin_centers.size))
    h_0 = np.zeros(energy_thresholds.size)

    for ii in range(energy_thresholds.size):
        d_min_idx_above_energy_threshold = d_min_idx[event_selection[ii, :] & same_event_ids]
        # h_2d[ii, :] = np.bincount(d_min_idx_above_energy_threshold, minlength=d_bin_centers.size)
        h_2d[ii, :] = np.bincount(d_min_idx_above_energy_threshold[d_min_idx_above_energy_threshold >= 0], minlength=d_bin_centers.size)

        h_0[ii] = np.sum(d_min_idx_above_energy_threshold < d_bin_edges[0]) / d_min_idx_above_energy_threshold.size

        # h_2d[ii, :] /= np.sum(h_2d[ii, :] * d_bin_widths)
        h_2d[ii, :] /= np.sum(h_2d[ii, :])

        # ax.stairs(h_2d[ii, :], edges=d_bin_edges, color=c_map(ii / (energy_thresholds.size - 1)))
        ax.stairs(h_2d[ii, :], edges=d_bin_edges, color=c_map(ii))

    mean_d_min = np.sum(h_2d * d_bin_centers[np.newaxis, :], axis=1)

    cmf = np.cumsum(h_2d, axis=1)
    median_d_min = np.apply_along_axis(lambda var: np.interp(0.5, var, d_bin_centers), 1, cmf)

    # fig, ax = plt.subplots()
    # ax.plot(energy_thresholds, mean_d_min)
    # ax.plot(energy_thresholds, median_d_min)
    # # ax.plot(d_bin_centers, cmf.T)
    # # ax.set_xscale('log')
    # plt.show()

    return median_d_min, h_0


def energy_threshold_variation_v2(energy_thresholds, event_selection, coincidences_struct, true, not_phantom_scattered):
    # ref = np.interp(200e-3, energy_thresholds.flatten(), yy.flatten())

    n_events = np.sum(event_selection, axis=1)
    # p_true = np.sum(event_selection & true[np.newaxis, :], axis=1) / n_events
    p_true = np.sum(event_selection & true[np.newaxis, :], axis=1) / n_events
    p_phantom_scatter = np.sum(event_selection & not_phantom_scattered[np.newaxis, :], axis=1) / n_events

    # fig, ax = plt.subplots()
    # ax.plot(energy_thresholds, n_events)
    #
    # ax_twin = ax.twinx()
    # ax_twin.plot(energy_thresholds, p_true)
    #
    # plt.show()

    return n_events, p_true, p_phantom_scatter


def check_event_ordering(coincidences_struct, multiplicity, group_indices, group_sizes):
    t1, t2 = coincidences_struct['time1'], coincidences_struct['time2']
    e1, e2 = coincidences_struct['energy1'], coincidences_struct['energy2']
    event_id1, event_id2 = coincidences_struct['eventID1'], coincidences_struct['eventID2']

    # Events are sorted according to the time of the first event
    # print(t1.size - np.sum(t1[:-1] <= t1[1:]))

    for ii in trange(group_indices.size - 1):
    # for ii in trange(100):
        idx_group = np.arange(group_indices[ii], group_indices[ii + 1])
        t1_group, t2_group = t1[idx_group], t2[idx_group]

        order_check = t2_group[:-1] <= t2_group[1:]

        order_check = order_check | (~ (np.diff(t1_group) == 0))

        # aaa = np.diff(t1_group) == 0
        # order_check[~aaa] = True

        # check if sorted

        if not np.all(order_check):
            print(multiplicity[ii])
            print(order_check)
            # print(aaa)
            # print(e1[idx_group])
            # print(e2[idx_group])
            print(t1_group)
            print(t2_group)
            print()





    return 0


def blur_time_and_reorder(coincidences_struct, multiplicity, group_indices, group_sizes, fwhm_ps):
    #
    sigma_s = fwhm_ps / 1e12 / (2 * np.sqrt(2 * np.log(2)))

    #
    t1 = coincidences_struct['time1']
    t2 = coincidences_struct['time2']

    coincidences_struct_blurred = coincidences_struct.copy()

    for ii in trange(group_indices.size - 1):
    # for ii in trange(100):
        idx_group = np.arange(group_indices[ii], group_indices[ii + 1])
        current_coincidence_struct = coincidences_struct[idx_group]
        tt = np.hstack((t1[idx_group], t2[idx_group]))

        tt_unique, idx_inv = np.unique(tt, return_inverse=True)
        tt_unique += np.random.normal(loc=0, scale=sigma_s, size=tt_unique.size)

        # if tt_unique.size != multiplicity[ii]:
        #     print('Error: multiplicity inconsistency!')

        tt_blurred = tt_unique[idx_inv]
        # tt1_blurred = tt_blurred[:idx_group.size]
        # tt2_blurred = tt_blurred[idx_group.size:]
        current_coincidence_struct['time1'] = tt_blurred[:idx_group.size]
        current_coincidence_struct['time2'] = tt_blurred[idx_group.size:]

        #
        _, idx_inv_blurred = np.unique(tt_blurred, return_inverse=True)

        order = np.reshape(idx_inv, (2, idx_group.size))
        order_blurred = np.reshape(idx_inv_blurred, (2, idx_group.size))

        if not np.all(order == order_blurred):
            # Swap within the coincidences
            intra_swaps_necessary = order_blurred[0, :] > order_blurred[1, :]
            if np.any(intra_swaps_necessary):
                current_coincidence_struct = intra_swap(current_coincidence_struct, intra_swaps_necessary)

            # Swap the coincidence order
            ttt_1 = current_coincidence_struct['time1']
            inter_swap_1_necessary = not np.all(ttt_1[:-1] <= ttt_1[1:])
            if inter_swap_1_necessary:
                current_coincidence_struct = inter_swap_1(current_coincidence_struct)

            #
            ttt_2 = current_coincidence_struct['time2']
            inter_swap_2_necessary = not np.all((ttt_2[:-1] <= ttt_2[1:]) | (~ (np.diff(current_coincidence_struct['time1']) == 0)))
            if inter_swap_2_necessary:
                current_coincidence_struct = inter_swap_2(current_coincidence_struct)


            _, idx_inv_blurred_swapped = np.unique(np.hstack((current_coincidence_struct['time1'], current_coincidence_struct['time2'])), return_inverse=True)
            order_blurred_swapped = np.reshape(idx_inv_blurred_swapped, (2, idx_group.size))

            # # print(order)
            # print(order_blurred)
            # # print(intra_swaps_necessary)
            # # print(inter_swap_1_necessary)
            # print(inter_swap_2_necessary)
            # print(order_blurred_swapped)
            # print()

        coincidences_struct_blurred[idx_group] = current_coincidence_struct
    return coincidences_struct_blurred


def intra_swap(coincidence_struct, intra_swaps_necessary):
    fields = np.array(coincidence_struct.dtype.names)
    fields = np.reshape(fields, (int(fields.size / 2), 2), order='F')

    for f1, f2 in fields:
        tmp = coincidence_struct[f1][intra_swaps_necessary].copy()
        coincidence_struct[f1][intra_swaps_necessary] = coincidence_struct[f2][intra_swaps_necessary]
        coincidence_struct[f2][intra_swaps_necessary] = tmp

    return coincidence_struct


def inter_swap_1(coincidence_struct):
    idx_sort = np.argsort(coincidence_struct['time1'])
    return coincidence_struct[idx_sort]


def inter_swap_2(coincidence_struct):
    t1 = coincidence_struct['time1']
    t2 = coincidence_struct['time2']
    # sub-divide
    group_edges = np.arange(coincidence_struct.size - 1)[np.diff(t1) > 0] + 1
    group_edges = np.insert(group_edges, 0, 0)
    group_edges = np.append(group_edges, coincidence_struct.size)

    perm = np.zeros(coincidence_struct.size, dtype=int)
    for ii in range(group_edges.size - 1):
        group_indices = np.arange(group_edges[ii], group_edges[ii + 1])
        perm[group_indices] = group_indices[np.argsort(t2[group_indices])]

    return coincidence_struct[perm]


def energy_based_selection(coincidence_struct, group_indices):
    e1, e2 = coincidence_struct['energy1'], coincidence_struct['energy2']
    e12 = np.vstack((e1, e2))
    e = e1 + e2

    selection = np.zeros(group_indices.size - 1, dtype=int)
    for ii in trange(group_indices.size - 1):
        idx_group = np.arange(group_indices[ii], group_indices[ii + 1])
        selection[ii] = idx_group[np.argmax(e[idx_group])]

        e12_group = e12[:, idx_group]
        candidates = e12_group == np.max(e12_group.flatten())
        candidates = np.any(candidates, axis=0)

        # selection[ii] = idx_group[candidates][np.argmax(e[idx_group][candidates])]

    return selection


if __name__ == "__main__":
    main()
