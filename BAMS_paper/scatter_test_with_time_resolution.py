"""
Check the scatter test taking the time resolution into account

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.constants import speed_of_light
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# Auxiliary functions
from CASToR.lut import read_lut_binary, read_lut_header
from utilities import load_gate_data, filter_true, filter_phantom_scattered, separate_into_detector_categories
from CASToR.root_to_cdf import extract_coincidence_data, visualize_crystal_attribution, visualize_lors, analyze_deviation
from CASToR.npy_to_cdf import get_castor_id2


def main():
    lut = read_lut_binary(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert_dz_1_mm.lut')
    lut_header = read_lut_header(sys.path[1] + '/CASToR/TB_J-PET_7th_gen_brain_insert_dz_1_mm.hscan')

    # time_resolution = 0
    # time_resolution = 200
    # time_resolution = 400
    time_resolution = 600
    coincidences_struct = load_gate_data(time_resolution, True)

    # Use only 10 %
    coincidences_struct = coincidences_struct[:int(coincidences_struct.size * 1)]

    """Get the CASToR ID"""
    (x1, y1, z1, t1, gantry_id1, rsector_id1, crystal_id1, layer_id1,
     x2, y2, z2, t2, gantry_id2, rsector_id2, crystal_id2, layer_id2, num_entries) = extract_coincidence_data(coincidences_struct)

    c1 = get_castor_id2(gantry_id1, rsector_id1, crystal_id1, layer_id1)
    c2 = get_castor_id2(gantry_id2, rsector_id2, crystal_id2, layer_id2)

    # subset = np.arange(num_entries)
    # subset = np.random.choice(num_entries, replace=False, size=min(100, num_entries))
    # visualize_crystal_attribution(x1[subset], y1[subset], z1[subset], lut[c1[subset]])
    # visualize_crystal_attribution(x2[subset], y2[subset], z2[subset], lut[c2[subset]])
    # visualize_lors(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], lut[c1[subset]], lut[c2[subset]])
    # analyze_deviation(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], c1[subset], c2[subset], lut, lut_header)
    # verify_tof_sign(sx[subset], sy[subset], sz[subset], lut[c1[subset]], lut[c2[subset]], delta_t[subset])

    # layer_id_grid = np.arange(16 * 330)
    # layer_id_grid = np.reshape(layer_id_grid, (16, 330), order='F')[:, ::10]
    # layer_id_grid = layer_id_grid.flatten()
    # # layer_id_test = np.array(512, ndmin=1)
    #
    # c_grid_blurred = get_castor_id2(np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), layer_id_grid)
    # c_grid = get_castor_id2(np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), np.zeros(layer_id_grid.size, dtype=int), layer_id_grid, blur_z=False)
    # # c_test = get_castor_id2(np.zeros(layer_id_test.size, dtype=int), np.zeros(layer_id_test.size, dtype=int), np.zeros(layer_id_test.size, dtype=int), layer_id_test)
    #
    # # _, zz = np.unravel_index(layer_id_test, (16, 330), order='F')
    # # print(zz)
    #
    # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    #
    # # ax0.scatter(lut[c_grid]['Posy'], lut[c_grid]['Posz'], c=layer_id_grid)
    # ax0.scatter(lut[c_grid]['Posz'], lut[c_grid]['Posy'], c=layer_id_grid)
    # ax0.set_aspect(1)
    # ax0.set_xticks([])
    # ax0.set_ylabel(r'$y$ [mm]')
    # ax0.set_title(r'Before $z$ blurring')
    # ax0.set_xlim(580, 920)
    # # ax.plot(lut[c_test]['Posy'], lut[c_test]['Posz'], 'x', color='tab:red')
    #
    # # ax1.scatter(lut[c_grid_blurred]['Posy'], lut[c_grid_blurred]['Posz'], c=layer_id_grid)
    # ax1.scatter(lut[c_grid_blurred]['Posz'], lut[c_grid_blurred]['Posy'], c=layer_id_grid)
    # ax1.set_aspect(1)
    # ax1.set_xlabel(r'$z$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    # ax1.set_title(r'After $z$ blurring')
    # ax1.set_xlim(580, 920)
    # plt.show()

    """Run the scatter test"""

    dt = t2 - t1

    true = filter_true(coincidences_struct)
    # not_phantom_scattered = filter_phantom_scattered(coincidences_struct)
    # true = true & not_phantom_scattered

    pass_adjacency_test = run_adjacency_test(gantry_id1, rsector_id1, gantry_id2, rsector_id2, minimum_sector_difference=2, vis=False)

    bin_edges = np.linspace(-300, 300, 500 + 1)
    h, h_true, h_tbtb, h_tbbi, h_bibi, threshold, h_threshold, pass_scatter_test = run_scatter_test(
        gantry_id1[pass_adjacency_test], gantry_id2[pass_adjacency_test],c1[pass_adjacency_test],
        c2[pass_adjacency_test], dt[pass_adjacency_test], true[pass_adjacency_test], lut, bin_edges)

    # np.savez(sys.path[0] + '/Scatter_test_plot/time_resolution_%d_ps_without_phantom.npz' % time_resolution,
    #          bin_edges=bin_edges, h=h, h_true=h_true, h_tbtb=h_tbtb, h_tbbi=h_tbbi, h_bibi=h_bibi,
    #          threshold=threshold, h_threshold=h_threshold)

    sys.exit()

    preselection = pass_adjacency_test.copy()
    preselection[pass_adjacency_test] = pass_scatter_test

    np.save(sys.path[0] + '/Preselection/time_resolution_%d_ps_with_phantom.npy' % time_resolution, preselection)

    return 0


def run_adjacency_test(gantry_id1, rsector_id1, gantry_id2, rsector_id2, minimum_sector_difference=2, vis=False):
    # Treat the TB-J-PET as a single scanner
    new_gantry_id1 = gantry_id1.copy()
    new_gantry_id1[new_gantry_id1 == 1] = 0
    new_gantry_id1[new_gantry_id1 == 2] = 1

    new_gantry_id2 = gantry_id2.copy()
    new_gantry_id2[new_gantry_id2 == 1] = 0
    new_gantry_id2[new_gantry_id2 == 2] = 1

    # Old and incorrect approach
    # ravel_id1 = np.ravel_multi_index((new_gantry_id1, rsector_id1), (2, 24), order='C')
    # ravel_id2 = np.ravel_multi_index((new_gantry_id2, rsector_id2), (2, 24), order='C')
    # id_diff = np.abs(ravel_id1 - ravel_id2)

    # New approach
    different_scanner = new_gantry_id1 != new_gantry_id2
    scanner_0 = (new_gantry_id1 == 0) & (new_gantry_id2 == 0)
    scanner_1 = (new_gantry_id1 == 1) & (new_gantry_id2 == 1)
    # print(gantry_id1.size - np.sum(different_scanner) - np.sum(scanner_0) - np.sum(scanner_1))   # check consistency

    abs_diff = np.zeros(gantry_id1.size, dtype=int)
    abs_diff[different_scanner] = -1
    d_0 = np.abs(rsector_id1[scanner_0] - rsector_id2[scanner_0])
    abs_diff[scanner_0] = np.minimum(d_0, 24 - d_0)
    d_1 = np.abs(rsector_id1[scanner_1] - rsector_id2[scanner_1])
    abs_diff[scanner_1] = np.minimum(d_1, 12 - d_1)

    if vis:
        # h = np.bincount(id_diff)
        edges = np.arange(-1, 13 + 1) - 1 / 2
        centers = (edges[:-1] + edges[1:]) / 2
        h2, _ = np.histogram(abs_diff, bins=edges)

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        # ax.bar(np.arange(h.size), h, width=0.8)
        ax.bar(centers, h2, width=0.8)
        ax.set_xticks(centers)
        ax.set_xlabel('Absolute sector difference')
        ax.set_ylabel('Count')
        plt.show()

    # pass_adjacency_test = id_diff >= minimum_sector_difference
    pass_adjacency_test = abs_diff >= minimum_sector_difference
    pass_adjacency_test[abs_diff < 0] = True

    print('Passing adjacency test: %1.2f %%.' % (np.sum(pass_adjacency_test) / pass_adjacency_test.size * 100))

    return pass_adjacency_test


def run_scatter_test(gantry_id1, gantry_id2, c1, c2, dt, true, lut, bin_edges):
    tbtb, tbbi, bibi = separate_into_detector_categories(gantry_id1, gantry_id2)

    lut1, lut2 = lut[c1], lut[c2]
    dr = np.sqrt((lut1['Posx'] - lut2['Posx']) ** 2 + (lut1['Posy'] - lut2['Posy']) ** 2 + (lut1['Posz'] - lut2['Posz']) ** 2) * 1e-3  # [m]
    scatter_test = (dr - dt * speed_of_light) * 100  # cm

    # Histograms
    h, _ = np.histogram(scatter_test, bins=bin_edges)
    h_true, _ = np.histogram(scatter_test[true], bins=bin_edges)
    h_tbtb, _ = np.histogram(scatter_test[true & tbtb], bins=bin_edges)
    h_tbbi, _ = np.histogram(scatter_test[true & tbbi], bins=bin_edges)
    h_bibi, _ = np.histogram(scatter_test[true & bibi], bins=bin_edges)

    # norm = np.sum(h)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    norm = np.trapz(h, x=bin_centers)
    h, h_true, h_tbtb, h_tbbi, h_bibi = h.astype(float) / norm, h_true.astype(float) / norm, h_tbtb.astype(float) / norm, h_tbbi.astype(float) / norm, h_bibi.astype(float) / norm

    # Determine the minimum between the zero peak and the peak attributable to the shortest LOR with a low time
    # difference, i.e. originating from the center (about 30 cm)
    # search_interval = (bin_centers >= 0) & (bin_centers <= 0.3)
    search_interval = (bin_centers >= 0) & (bin_centers <= 0.3 * 100)
    idx_min = np.argmin(h[search_interval])
    threshold = bin_centers[search_interval][idx_min]
    h_threshold = h[search_interval][idx_min]

    if threshold > 10.:
        print('Lowering threshold from %1.1f cm to 10 cm.' % threshold)
        threshold = 10.

    pass_scatter_test = scatter_test > threshold
    print('Passing scatter test: %1.2f %%.' % (np.sum(pass_scatter_test) / pass_scatter_test.size * 100))

    return h, h_true, h_tbtb, h_tbbi, h_bibi, threshold, h_threshold, pass_scatter_test


def plot_scatter_test():
    time_res = ['0', '200', '400', '600']
    line_styles = ['-', '-.', '--', ':']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax_sub = fig.add_axes([0.55, 0.475, 0.34, 0.35])

    for ii in range(len(time_res)):
        npz_data = np.load(sys.path[0] + '/Scatter_test_plot/time_resolution_%s_ps_without_phantom.npz' % time_res[ii])
        # npz_data = np.load(sys.path[0] + '/Scatter_test_plot/time_resolution_%s_ps_with_phantom.npz' % time_res[ii])
        # npz_data = np.load(sys.path[0] + '/Scatter_test_plot/time_resolution_%s_ps_with_phantom_true_in_detector.npz' % time_res[ii])
        bin_edges = npz_data['bin_edges']
        h, h_true, h_tbtb, h_tbbi, h_bibi = npz_data['h'], npz_data['h_true'], npz_data['h_tbtb'], npz_data['h_tbbi'], npz_data['h_bibi']
        threshold, h_threshold = npz_data['threshold'], npz_data['h_threshold']

        ax.stairs(h, edges=bin_edges, color='black', linestyle=line_styles[ii])
        ax.stairs(h_true, edges=bin_edges, color='tab:blue', linestyle=line_styles[ii])
        ax.stairs(h_tbtb, edges=bin_edges, color='tab:orange', linestyle=line_styles[ii])
        ax.stairs(h_tbbi, edges=bin_edges, color='tab:green', linestyle=line_styles[ii])
        ax.stairs(h_bibi, edges=bin_edges, color='tab:red', linestyle=line_styles[ii])
        ax.plot([threshold, threshold], [0, h_threshold], color='k', linestyle=line_styles[ii])

        ax_sub.stairs(h, edges=bin_edges, color='black', linestyle=line_styles[ii])
        ax_sub.stairs(h_true, edges=bin_edges, color='tab:blue', linestyle=line_styles[ii])
        ax_sub.stairs(h_tbtb, edges=bin_edges, color='tab:orange', linestyle=line_styles[ii])
        ax_sub.stairs(h_tbbi, edges=bin_edges, color='tab:green', linestyle=line_styles[ii])
        ax_sub.stairs(h_bibi, edges=bin_edges, color='tab:red', linestyle=line_styles[ii])
        ax_sub.plot([threshold, threshold], [0, h_threshold], color='k', linestyle=line_styles[ii])

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        print('%s ps:' % time_res[ii])
        print('TB-BI: %1.2f %%' % (np.sum(h_tbbi[bin_centers >= threshold]) / np.sum(h_tbbi) * 100))
        print('BI-BI: %1.2f %%\n' % (np.sum(h_bibi[bin_centers >= threshold]) / np.sum(h_bibi) * 100))


    ax.set_xlim(-100, 300)
    ax.set_ylim(0, 0.014)
    # ax.set_ylim(0, 0.01)
    # ax.set_ylim(0, 0.01)
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

    # ax.annotate(r'$\bf{All}$' '\n' r'$\bf{coinc.}$', xy=(70, 1e-2), xytext=(10, 1e-2), ha='left', va='center',
    #             arrowprops=dict(arrowstyle='->', color='k'))
    ax.annotate(r'$\bf{All}$' '\n' r'$\bf{coinc.}$', xy=(70, 0.9e-2), xytext=(10, 0.9e-2), ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='k'))

    ax_sub.set_xlim(-5, 55)
    ax_sub.set_ylim(0, 6e-4)
    # ax_sub.set_ylim(0, 2.4e-4)
    # ax_sub.set_ylim(0, 8e-4)

    ax_sub.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax_sub.set_yscale('log')
    # ax_sub.set_ylim(1e-6, 1e-1)
    ax_sub.set_xticks([0, 25, 50])
    # ax_sub.set_yticks([0, 1e-4, 2e-4])
    # ax_sub.set_yticks([0, 2e-4, 4e-4, 6e-4, 8e-4])

    plt.show()

    return 0


if __name__ == "__main__":
    # main()
    plot_scatter_test()
