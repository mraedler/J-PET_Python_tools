"""
Cast GATE root output file to binary cdf file of CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
from os.path import basename, splitext
import numpy as np
from uproot import open as open_root
from scipy.optimize import curve_fit
from scipy.constants import speed_of_light
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Auxiliary functions
from CASToR.lut import read_lut_binary
from CASToR.utilities import get_gantry_filter, get_true_filter
# sys.path.append('../')
from data_structures import load_or_convert_to_structured_array


def main():
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-04-11_10-03-38/results.root')  # TB only: 10 s
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-03-12_15-12-50/results.root')  # TB-B: 10 s
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-05-28_15-27-04/results.root')  # TB-B: 100 s
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-06-09_12-04-35/results.root')  # TB-B_2: 100 s
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo/2024-07-30_09-13-38/results.root')  # TB-B: 100 s Derenzo
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo/2024-08-05_18-05-27/results.root')  # TB-B: 100 s Derenzo non-collinearity
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo_3/2024-10-17_09-00-35/results.root')  # TB-B: 100 s Derenzo_3
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo_3/2024-10-17_11-24-19/results.root')  # TB-B: 1000 s Derenzo_3
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo_3_axial_3mm/2024-10-30_09-53-15/results.root')  # TB-B: 100 s Derenzo_3_axial_3mm
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Derenzo_3_axial_3mm_B2/2024-11-06_18-25-48/results.root')  # TB-B: 100 s Derenzo_3_axial_10mm
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Hexagon/2024-12-04_10-26-46/results.root')  # TB-B: 100 s Hexagon
    # root_file = open_root('/home/martin/J-PET/Gate_Output/Hexagon/2024-12-06_18-34-44/results.root')  # TB-B: 1000 s Hexagon
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Output/2025-12-09_20-03-25/merged.root')  # TB-B: 100 s Derenzo_3
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Output/2025-12-09_20-03-25/merged2.root')  # TB-B: 100 s Derenzo_3
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Output/2025-12-09_20-03-25/merged2.root')  # TB-B: 100 s Derenzo_3
    root_file = open_root('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Output/2025-12-09_21-51-31/merged.root')  # TB-B: 100 s Derenzo_3


    necessary_keys = ['globalPosX1', 'globalPosY1', 'globalPosZ1', 'time1', 'gantryID1', 'rsectorID1', 'crystalID1', 'layerID1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonCrystal1', 'eventID1', 'energy1',
                      'globalPosX2', 'globalPosY2', 'globalPosZ2', 'time2', 'gantryID2', 'rsectorID2', 'crystalID2', 'layerID2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'comptonCrystal2', 'eventID2', 'energy2']
    # necessary_keys = ['globalPosX1', 'globalPosY1', 'globalPosZ1', 'time1', 'gantryID1', 'rsectorID1', 'crystalID1', 'layerID1',
    #                   'globalPosX2', 'globalPosY2', 'globalPosZ2', 'time2', 'gantryID2', 'rsectorID2', 'crystalID2', 'layerID2']
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'], keys=necessary_keys, overwrite=False)

    # plot_energy_histogram(coincidences_struct['energy1'], coincidences_struct['energy2'])

    # Apply filters
    gantry_filter, gf_tag = get_gantry_filter(coincidences_struct, 'TOT')  # TOT, TBTB, TBB, BB
    true_filter, tf_tag = get_true_filter(coincidences_struct, filter_true=False)
    coincidences_struct = coincidences_struct[gantry_filter & true_filter]

    (x1, y1, z1, t1, gantry_id1, rsector_id1, crystal_id1, layer_id1,
     x2, y2, z2, t2, gantry_id2, rsector_id2, crystal_id2, layer_id2, num_entries) = extract_coincidence_data(coincidences_struct)

    # sx1, sy1, sz1 = coincidences_struct['sourcePosX1'], coincidences_struct['sourcePosY1'], coincidences_struct['sourcePosZ1']
    # sx2, sy2, sz2 = coincidences_struct['sourcePosX2'], coincidences_struct['sourcePosY2'], coincidences_struct['sourcePosZ2']
    # sx, sy, sz = (sx1 + sx2) / 2, (sy1 + sy2) / 2, (sz1 + sz2) / 2
    # print(np.sum(sx1 == sx2) / sx1.size)

    #
    delta_t = t1 - t2  # [s]
    tof_flag = False

    if tof_flag:
        tof_fwhm = 300  # [ps]
        tof_tag = '_%d_ps' % tof_fwhm
        tof_sigma = tof_fwhm * 1e-12 / (2 * np.sqrt(2 * np.log(2)))  # [s]
        delta_t += np.random.normal(loc=0., scale=tof_sigma, size=delta_t.size)
    else:
        tof_fwhm = 'inf'  # [ps]
        tof_tag = '_inf_ps'

    # Load the LUT
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_Brain.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_Brain_2.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_axial_3mm_Brain.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_axial_6mm_Brain.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_6th_gen_7_rings_gap_2cm_axial_10mm_Brain.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_JPET_axial_3mm_B2_axial_3mm.lut'
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_6_30_3_BI_4_18_3.lut'
    lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_J-PET_7th_gen_brain_insert_dz_1_mm.lut'

    lut = read_lut_binary(lut_path)

    show_figures = True

    # Allocate
    t = np.zeros(num_entries, dtype=np.uint32)
    tof = np.zeros(num_entries, dtype=np.float32)
    c1 = np.zeros(num_entries, dtype=np.uint32)
    c2 = np.zeros(num_entries, dtype=np.uint32)

    # Events to be processed
    ii_tbp = np.arange(num_entries)
    if show_figures:
        ii_tbp = np.random.choice(num_entries, replace=False, size=100000)

    # Loop over events
    for ii in tqdm(ii_tbp):
        # Time in milliseconds
        time_ms = np.round(1000 * (t1[ii] + t2[ii]) / 2).astype(np.uint32)
        tof_ps = (delta_t[ii] / 1e-12).astype(np.float32)

        # Crystal index
        castor_id1 = get_castor_id(gantry_id1[ii], rsector_id1[ii], crystal_id1[ii], layer_id1[ii])
        castor_id2 = get_castor_id(gantry_id2[ii], rsector_id2[ii], crystal_id2[ii], layer_id2[ii])

        t[ii] = time_ms
        tof[ii] = tof_ps
        c1[ii] = castor_id1
        c2[ii] = castor_id2

        # print(x1[ii], -y1[ii], z1[ii])
        # print(lut[kk]['Posx'], lut[kk]['Posy'], lut[kk]['Posz'])
        # print()

    if show_figures:
        # visualize_crystal_attribution(x1[ii_tbp], y1[ii_tbp], z1[ii_tbp], lut[c1[ii_tbp]])
        # visualize_crystal_attribution(x2[ii_tbp], y2[ii_tbp], z2[ii_tbp], lut[c2[ii_tbp]])
        # visualize_lors(x1[ii_tbp], y1[ii_tbp], z1[ii_tbp], x2[ii_tbp], y2[ii_tbp], z2[ii_tbp], lut[c1[ii_tbp]], lut[c2[ii_tbp]])
        analyze_deviation(x1[ii_tbp], y1[ii_tbp], z1[ii_tbp], x2[ii_tbp], y2[ii_tbp], z2[ii_tbp], lut[c1[ii_tbp]], lut[c2[ii_tbp]])
        # verify_tof_sign(sx[ii_tbp], sy[ii_tbp], sz[ii_tbp], lut[c1[ii_tbp]], lut[c2[ii_tbp]], delta_t[ii_tbp])

        sys.exit()

    # Initialize the structured array
    if tof_flag:
        d_type = np.dtype([('t',  np.uint32), ('tof',  np.float32), ('c1', np.uint32), ('c2', np.uint32)])
    else:
        d_type = np.dtype([('t', np.uint32), ('c1', np.uint32), ('c2', np.uint32)])

    cdf_file = np.empty(num_entries, dtype=d_type)
    cdf_file['t'] = t
    if tof_flag:
        cdf_file['tof'] = tof
    cdf_file['c1'] = c1
    cdf_file['c2'] = c2

    # Save as binary file
    cdf_file_path = 'TB_6_30_3_BI_4_18_3_' + gf_tag + tf_tag + tof_tag + '.cdf'
    binary_file = open(cdf_file_path, 'wb')
    binary_file.write(cdf_file.tobytes())
    binary_file.close()

    # Write the header file
    t_i = np.floor(np.min(t) / 1000).astype(int)
    t_f = np.ceil(np.max(t) / 1000).astype(int)
    tof_range = np.ceil((np.max(delta_t) - np.min(delta_t)) / 1e-12).astype(int)
    castor_data_header_file(lut_path, cdf_file_path, num_entries, t_i, t_f, tof_flag, tof_fwhm, tof_range)

    return 0


def plot_energy_histogram(energy_1, energy_2):
    energy = np.concatenate((energy_1, energy_2)) * 1e3  # [keV]

    energy_bins = np.linspace(200, 450, 100)
    energy_centers = (energy_bins[1:] + energy_bins[:-1]) / 2
    energy_widths = energy_bins[1:] - energy_bins[:-1]

    h, _ = np.histogram(energy, bins=energy_bins)

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(energy_centers, h, width=energy_widths)
    ax.set_xlim(energy_bins[0], energy_bins[-1])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel('Energy deposition [keV]')
    ax.set_ylabel('Counts')
    plt.show()

    return 0


def extract_coincidence_data(coincidences):
    x1 = coincidences['globalPosX1']
    y1 = coincidences['globalPosY1']
    z1 = coincidences['globalPosZ1']
    t1 = coincidences['time1']
    gantry_id1 = coincidences['gantryID1']
    rsector_id1 = coincidences['rsectorID1']
    crystal_id1 = coincidences['crystalID1']
    layer_id1 = coincidences['layerID1']

    x2 = coincidences['globalPosX2']
    y2 = coincidences['globalPosY2']
    z2 = coincidences['globalPosZ2']
    t2 = coincidences['time2']
    gantry_id2 = coincidences['gantryID2']
    rsector_id2 = coincidences['rsectorID2']
    crystal_id2 = coincidences['crystalID2']
    layer_id2 = coincidences['layerID2']
    num_entries = coincidences.size

    return (x1, y1, z1, t1, gantry_id1, rsector_id1, crystal_id1, layer_id1,
            x2, y2, z2, t2, gantry_id2, rsector_id2, crystal_id2, layer_id2, num_entries)


def get_castor_id(gantry_id, rsector_id, crystal_id, layer_id):
    # gantry_0_shape = np.array([2, 7, 24, 16 * 110], dtype=np.uint32)
    # gantry_1_shape = np.array([2, 1, 10, 16 * 110], dtype=np.uint32)

    # gantry_0_shape = np.array([2, 7, 24, 16 * 55], dtype=np.uint32)
    # gantry_1_shape = np.array([2, 1, 10, 16 * 55], dtype=np.uint32)

    # gantry_0_shape = np.array([2, 7, 24, 16 * 33], dtype=np.uint32)
    # gantry_1_shape = np.array([2, 1, 10, 16 * 33], dtype=np.uint32)

    gantry_0_shape = np.array([2, 7, 24, 16 * 110], dtype=np.uint32)
    gantry_1_shape = np.array([2, 1, 10, 23 * 110], dtype=np.uint32)

    # Follow the CASToR hierarchy
    ring_idx = crystal_id
    rsector_idx = rsector_id

    if gantry_id == 0:
        layer_number = np.floor(layer_id / gantry_0_shape[3]).astype(int)
        layer_idx = layer_id - layer_number * gantry_0_shape[3]
        # rsector_idx = np.mod(24 - rsector_idx + 12, 24)
        rsector_idx = np.mod(gantry_0_shape[2] - rsector_idx + (gantry_0_shape[2] / 2).astype(int), gantry_0_shape[2])

        castor_id = np.ravel_multi_index((layer_number, ring_idx, rsector_idx, layer_idx), gantry_0_shape, order='C').astype(np.uint32)

    elif gantry_id == 1:
        # layer_number = np.floor(layer_id / (23 * 110)).astype(int)
        # layer_idx = layer_id - layer_number * 23 * 110
        # rsector_idx = np.mod(10 - rsector_idx + 5, 10)
        # castor_id = np.ravel_multi_index((layer_number, ring_idx, rsector_idx, layer_idx), (2, 1, 10, 23 * 110), order='C').astype(np.uint32) + offset

        layer_number = np.floor(layer_id / gantry_1_shape[3]).astype(int)
        layer_idx = layer_id - layer_number * gantry_1_shape[3]
        rsector_idx = np.mod(10 - rsector_idx + 5, 10)

        castor_id = np.ravel_multi_index((layer_number, ring_idx, rsector_idx, layer_idx), gantry_1_shape, order='C').astype(np.uint32) + np.prod(gantry_0_shape)

    else:
        sys.exit('Error')

    return castor_id


def visualize_crystal_attribution(x_int, y_int, z_int, lut):
    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=(1, 2.5), sharey=True)

    ax0.plot(x_int, y_int, '.', color='tab:blue')
    ax0.plot(lut['Posx'], lut['Posy'], '.', color='tab:orange')

    ax1.plot(z_int, x_int, '.', color='tab:blue', label='Interaction position')
    ax1.plot(lut['Posz'], lut['Posx'], '.', color='tab:orange', label='Crystal centers')
    # ax1.plot(z_int, y_int, '.', color='tab:blue', label='Interaction position')
    # ax1.plot(lut['Posz'], lut['Posy'], '.', color='tab:orange', label='Crystal centers')

    ax0.set_xlim(-500, 500)
    ax0.set_ylim(-500, 500)
    ax0.set_aspect(1)
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')

    ax1.set_xlim(-1215, 1215)
    # ax1.set_ylim(-500, 500)
    ax1.set_aspect(1)
    ax1.set_xlabel(r'$z$ [mm]')
    ax1.set_ylabel(r'$y$ [mm]')
    ax1.legend()

    plt.show()

    return 0


def visualize_lors(x_1, y_1, z_1, x_2, y_2, z_2, lut_1, lut_2):
    y_sign = 1
    # y_sign = -1

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=(1, 2.5), sharey=True)

    x_12 = np.stack((x_1, x_2), axis=1).T
    y_12 = np.stack((y_1, y_2), axis=1).T
    z_12 = np.stack((z_1, z_2), axis=1).T

    lut_x_12 = np.stack((lut_1['Posx'], lut_2['Posx']), axis=1).T
    lut_y_12 = y_sign * np.stack((lut_1['Posy'], lut_2['Posy']), axis=1).T
    lut_z_12 = np.stack((lut_1['Posz'], lut_2['Posz']), axis=1).T

    # ax0.plot(x_12, y_12)
    ax0.plot(lut_x_12, lut_y_12)

    # ax1.plot(z_12, y_12)
    ax1.plot(lut_z_12, lut_y_12)

    ax0.set_xlim(-500, 500)
    ax0.set_ylim(-500, 500)
    ax0.set_aspect(1)
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')

    ax1.set_xlim(-1215, 1215)
    # ax1.set_ylim(-500, 500)
    ax1.set_aspect(1)
    ax1.set_xlabel(r'$z$ [mm]')
    ax1.set_ylabel(r'$y$ [mm]')

    plt.show()
    return 0


def analyze_deviation(x_1, y_1, z_1, x_2, y_2, z_2, c_1, c_2, lut, lut_header):
    # Number of layers per scanner (number of layers to be grouped)
    # n_g = 2
    n_g = 1

    # From the LUT header
    n_crystals_per_layer = lut_header['number of crystals in layer']
    n_layers = n_crystals_per_layer.size
    half_crystals_size_depth = lut_header['crystals size depth'] / 2
    half_crystals_size_transaxial = lut_header['crystals size transaxial'] / 2
    half_crystals_size_axial = lut_header['crystals size axial'] / 2

    lut_1, lut_2 = lut[c_1], lut[c_2]
    c_12 = np.concatenate((c_1, c_2))

    # Rotate both the assigned crystal locations and the interaction position back to the first crystal location
    # to better compare them
    vx_1, vy_1 = lut_1['OrVx'], lut_1['OrVy']
    lut_d_1 = lut_1['Posx'] * vx_1 + lut_1['Posy'] * vy_1
    lut_l_1 = lut_1['Posx'] * vy_1 - lut_1['Posy'] * vx_1
    d_1 = x_1 * vx_1 + y_1 * vy_1
    l_1 = x_1 * vy_1 - y_1 * vx_1

    vx_2, vy_2 = lut_2['OrVx'], lut_2['OrVy']
    lut_d_2 = lut_2['Posx'] * vx_2 + lut_2['Posy'] * vy_2
    lut_l_2 = lut_2['Posx'] * vy_2 - lut_2['Posy'] * vx_2
    d_2 = x_2 * vx_2 + y_2 * vy_2
    l_2 = x_2 * vy_2 - y_2 * vx_2

    # Merge both interaction positions
    d_12, l_12 = np.concatenate((d_1, d_2)), np.concatenate((l_1, l_2))
    lut_d_12, lut_l_12 = np.concatenate((lut_d_1, lut_d_2)), np.concatenate((lut_l_1, lut_l_2))

    # The depth coordinate of the crystals should now be unique, only differing by the radius of the rings
    ring_radii = np.unique(np.round(lut_d_12, decimals=3))

    if ring_radii.size > n_layers:
        sys.exit('More ring radii than expected: % s' % ring_radii)

    # Deviation between interaction points and assigned crystal locations
    dd = d_12 - lut_d_12
    dl = l_12 - lut_l_12
    dz = np.concatenate((z_1 - lut_1['Posz'], z_2 - lut_2['Posz']))

    #
    layer_indices = np.insert(np.cumsum(n_crystals_per_layer), 0, 0)
    n_scanners = np.ceil(n_layers / n_g).astype(int) + 1

    fig, axes = plt.subplots(n_scanners, 3, figsize=(12, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ii in range(n_layers):
        selection = (c_12 >= layer_indices[ii]) & (c_12 < layer_indices[ii + 1])

        dd_edges, dd_centers, dd_widths = symmetric_binning_with_excess(100, half_crystals_size_depth[ii], p=0.2)
        h, _ = np.histogram(dd[selection], bins=dd_edges, density=False)
        if np.sum(selection) - np.sum(h) > 0:
            print('Warning: not all events are contained in the histogram')

        def fit_function(var, alpha, beta): return beta * skewed_box(var, alpha, half_width=half_crystals_size_depth[ii])
        p_opt, p_cov = curve_fit(fit_function, dd_centers, h, p0=[0., 1.])
        p_err = np.sqrt(np.diag(p_cov))
        # print('α = %1.3f ± %1.3f' % (p_opt[0], p_err[0]))
        print('Δ = %1.3f' % (half_crystals_size_depth[ii] * (1 + 2 * p_opt[0] / 3)))

        axes[ii // n_g, 0].stairs(h, edges=dd_edges, color=colors[ii], label='Ring %d' % ii)
        axes[ii // n_g, 0].plot(dd_centers, fit_function(dd_centers, *p_opt), color=colors[ii])
        axes[ii // n_g, 0].plot(dd_centers, p_opt[1] * (1 - 2 * p_opt[0]) / (2 * half_crystals_size_depth[ii]) * np.exp(- 0.00965 * (dd_centers + half_crystals_size_depth[ii])), linestyle='--', color=colors[ii])

        dl_edges, _, _ = symmetric_binning_with_excess(100, half_crystals_size_transaxial[ii], p=0.1)
        h, _ = np.histogram(dl[selection], bins=dl_edges)
        if np.sum(selection) - np.sum(h) > 0:
            print('Warning: not all events are contained in the histogram')
        axes[ii // n_g, 1].stairs(h, edges=dl_edges, color=colors[ii])

        dz_edges, _, _ = symmetric_binning_with_excess(100, half_crystals_size_axial[ii], p=0.1)
        h, _ = np.histogram(dz[selection], bins=dz_edges)
        if np.sum(selection) - np.sum(h) > 0:
            print('Warning: not all events are contained in the histogram')
        axes[ii // n_g, 2].stairs(h, edges=dz_edges, color=colors[ii])

    #
    np.vectorize(lambda x: x.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)))(axes)

    axes[0, 0].set_title('Longitudinal (depth)')
    axes[0, 1].set_title('Lateral')
    axes[0, 2].set_title('Axial')

    axes[-1, 0].set_xlabel(r'$d$ [mm]')
    axes[-1, 1].set_xlabel(r'$\ell$ [mm]')
    axes[-1, 2].set_xlabel(r'$z$ [mm]')

    for ii in range(n_scanners):
        fig.text(0.01, (axes[ii, 0].get_position().y0 + axes[ii, 0].get_position().y1) / 2, 'Scanner %d' % ii, ha='left', va='center', fontsize=12, fontweight='bold')

    plt.show()

    return 0


def skewed_box(x, alpha, half_width):
    y = np.zeros(x.shape)
    inside = np.abs(x) <= half_width
    y[inside] = 1 / (2 * half_width) + alpha * x[inside] / half_width ** 2
    return y


def symmetric_binning_with_excess(n, e, p=0.2):
    # n must be odd
    if n % 2 == 0:
        n += 1

    n_in_half = np.round(n * (1 - p) / 2).astype(int)
    n_out_half = ((n - (2 * n_in_half - 1)) / 2).astype(int)

    samples_in_half = np.linspace(0, e, n_in_half)
    samples_out_half = samples_in_half[-1] + np.arange(1, n_out_half + 1) * (samples_in_half[1] - samples_in_half[0])

    samples = np.concatenate((-np.flip(samples_out_half), -np.flip(samples_in_half), samples_in_half[1:], samples_out_half))

    edges = samples
    centers = (edges[1:] + edges[:-1]) / 2
    widths = edges[1:] - edges[:-1]

    return edges, centers, widths


def verify_tof_sign(source_x, source_y, source_z, crystal_1, crystal_2, delta_t):
    y_sign = 1
    # y_sign = -1

    source_crystal_1_distance = np.sqrt((crystal_1['Posx'] - source_x) ** 2 + (y_sign * crystal_1['Posy'] - source_y) ** 2 + (crystal_1['Posz'] - source_z) ** 2)
    source_crystal_2_distance = np.sqrt((crystal_2['Posx'] - source_x) ** 2 + (y_sign * crystal_2['Posy'] - source_y) ** 2 + (crystal_2['Posz'] - source_z) ** 2)
    delta_d = (source_crystal_1_distance - source_crystal_2_distance) / speed_of_light / 1e-6  # [ns]
    delta_t /= 1e-9  # [ns]

    t_edges = np.linspace(-3, 1, 101)  # [ns]
    h, _, _ = np.histogram2d(delta_t, delta_d, bins=[t_edges, t_edges])
    h = h.astype(int)

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.stairs(np.sum(h, axis=1), edges=t_edges)
    ax0.set_xlim(t_edges[0], t_edges[-1])
    # ax0.set_ylim(0, 3.25e4)
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax0.set_xlabel(r'$t_1-t_2$ [ns]')
    ax0.set_ylabel('Counts')

    # im = ax1.imshow(h.T, origin='lower', extent=[t_edges[0], t_edges[-1], t_edges[0], t_edges[-1]])
    im = ax1.imshow(h.T, origin='lower', extent=[t_edges[0], t_edges[-1], t_edges[0], t_edges[-1]], norm=LogNorm(vmin=1, vmax=np.max(h)))
    # im = ax1.imshow(h.T, origin='lower', extent=[t_edges[0], t_edges[-1], t_edges[0], t_edges[-1]], norm=LogNorm(vmin=1, vmax=1.5e4))
    cax = make_axes_locatable(ax1).append_axes('right', size='4%', pad=0.05)
    c_bar = fig.colorbar(im, cax=cax, orientation='vertical', label='Counts')
    # c_bar.formatter.set_powerlimits((0, 0))
    ax1.plot([t_edges[0], t_edges[-1]], [t_edges[0], t_edges[-1]], linestyle='--', color='tab:red')
    ax1.set_yticks([-3, -2, -1, 0, 1])
    ax1.set_xlabel(r'$t_1-t_2$ [ns]')
    ax1.set_ylabel(r'$(|\mathbf{S}-\mathbf{C}_1|-|\mathbf{S}-\mathbf{C}_2|)/c_0$ [ns]')

    plt.show()

    return 0


def castor_data_header_file(lut_path, data_file_path, number_of_events, t_i, t_f, tof_flag, tof_fwhm, tof_range):
    mandatory_fields = {'Scanner name': splitext(basename(lut_path))[0],
                        'Data filename': data_file_path,
                        'Data type': 'PET',
                        'Data mode': 'list-mode',  # histogram, normalization
                        'Isotope': 'unknown',
                        'Number of events': str(number_of_events),
                        'Start time (s)': str(t_i),
                        'Duration (s)': str(t_f - t_i),
                        'Maximum number of lines per event': '1',
                        'Calibration factor': '1',
                        'Attenuation correction flag': '0',
                        'Normalization correction flag': '0',
                        'Scatter correction flag': '0',
                        'Random correction flag': '0',
                        'TOF information flag': str(int(tof_flag)),
                        'TOF resolution (ps)': str(tof_fwhm),
                        'Per event TOF resolution flag': '0',
                        'List TOF measurement range (ps)': str(tof_range)}

    cdh_file = open(splitext(data_file_path)[0] + '.cdh', 'w')
    for key, value in mandatory_fields.items():
        cdh_file.write(key + ': ' + value + '\n')
    cdh_file.close()

    # todo: add optional fields
    optional_fields = {'Maximum axial difference mm': '-1, -1, -1, -1',  # [mm] only for sensitivity image computation for list-mode data and when no normalization file is provided
                       'Histo TOF number of bins': '10',  # required if TOF information flag is on for histogram data
                       'Histo TOF bin size (ps)': '10'}  # required if TOF information flag is on for histogram data
    return 0


if __name__ == "__main__":
    main()
