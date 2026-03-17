"""
Generate the binary LUT file for CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import erf
from uproot import open as open_root
from tqdm import tqdm

# Auxiliary functions
from CASToR.lut import read_lut_binary, read_lut_header
from CASToR.root_to_cdf import extract_coincidence_data, visualize_crystal_attribution, analyze_deviation, visualize_lors, verify_tof_sign, castor_data_header_file
from CASToR.utilities import get_true_filter
from data_structures import load_or_convert_to_structured_array


def main():
    use_tof = False
    # lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/TB_J-PET_7th_gen_brain_insert_dz_1_mm.lut'
    lut_path = '/home/martin/J-PET/CASToR/castor/config/scanner/Modular_scanner.lut'
    lut = read_lut_binary(lut_path)
    # sys.exit()
    lut_header = read_lut_header('/home/martin/J-PET/CASToR/castor/config/scanner/Modular_scanner.hscan')

    # root_file = open_root('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-03-11_11-08-35/output.root')  # New TB-J-PET
    # root_file = open_root('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-03-11_14-38-36/output.root')  # New TB-J-PET
    # root_file = open_root('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-03-20_16-51-20/output.root')  # New TB-J-PET + brain insert
    # root_file = open_root('/home/martin/J-PET/Gate_Output/New_TB-J-PET/2025-03-20_17-43-03/output.root')  # New TB-J-PET + brain insert
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.4/New_TB_J-PET_Brain/Output/2025-12-09_21-51-31/merged.root')  # Derenzo_3
    root_file = open_root('/home/martin/J-PET/pnt_1.root')  # Ermias test

    necessary_keys = ['globalPosX1', 'globalPosY1', 'globalPosZ1', 'time1', 'gantryID1', 'rsectorID1', 'crystalID1', 'layerID1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonCrystal1', 'RayleighCrystal1', 'eventID1',
                      'globalPosX2', 'globalPosY2', 'globalPosZ2', 'time2', 'gantryID2', 'rsectorID2', 'crystalID2', 'layerID2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2', 'comptonCrystal2', 'RayleighCrystal2', 'eventID2']
    # coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'], keys=necessary_keys, overwrite=False)
    coincidences_struct = load_or_convert_to_structured_array(root_file['Coincidences'], keys=necessary_keys, overwrite=False)

    true_filter, tf_tag = get_true_filter(coincidences_struct, filter_true=False)
    # print(np.sum(true_filter) / true_filter.size)
    # print(tf_tag)

    # todo: add further filters

    (x1, y1, z1, t1, gantry_id1, rsector_id1, crystal_id1, layer_id1,
     x2, y2, z2, t2, gantry_id2, rsector_id2, crystal_id2, layer_id2, num_entries) = extract_coincidence_data(coincidences_struct)

    sx1, sy1, sz1 = coincidences_struct['sourcePosX1'], coincidences_struct['sourcePosY1'], coincidences_struct['sourcePosZ1']
    sx2, sy2, sz2 = coincidences_struct['sourcePosX2'], coincidences_struct['sourcePosY2'], coincidences_struct['sourcePosZ2']
    sx, sy, sz = (sx1 + sx2) / 2, (sy1 + sy2) / 2, (sz1 + sz2) / 2

    c1 = get_castor_id2(gantry_id1, rsector_id1, crystal_id1, layer_id1, blur_z=False)
    c2 = get_castor_id2(gantry_id2, rsector_id2, crystal_id2, layer_id2, blur_z=False)
    time_ms = np.round(1000 * (t1 + t2) / 2).astype(np.uint32)
    delta_t_ps = ((t1 - t2) / 1e-12).astype(np.float32)

    if use_tof:
        tof_fwhm_ps = 500  # [ps]
        # tof_sigma_ps = tof_fwhm_ps / (2 * np.sqrt(2 * np.log(2)))  # [s]
        # delta_t_ps += np.random.normal(loc=0., scale=tof_sigma_ps, size=delta_t_ps.size).astype(np.float32)

        # Initialize the structured array
        d_type = np.dtype([('t',  np.uint32), ('tof',  np.float32), ('c1', np.uint32), ('c2', np.uint32)])
    else:
        tof_fwhm_ps = 0
        d_type = np.dtype([('t', np.uint32), ('c1', np.uint32), ('c2', np.uint32)])

    subset = np.arange(num_entries)
    # subset = np.random.choice(num_entries, replace=False, size=min(100, num_entries))
    # visualize_crystal_attribution(x1[subset], y1[subset], z1[subset], lut[c1[subset]])
    # visualize_crystal_attribution(x2[subset], y2[subset], z2[subset], lut[c2[subset]])
    # visualize_lors(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], lut[c1[subset]], lut[c2[subset]])
    analyze_deviation(x1[subset], y1[subset], z1[subset], x2[subset], y2[subset], z2[subset], c1[subset], c2[subset], lut, lut_header)
    # verify_tof_sign(sx[subset], sy[subset], sz[subset], lut[c1[subset]], lut[c2[subset]], delta_t[subset])

    cdf_file = np.empty(num_entries, dtype=d_type)
    cdf_file['t'] = time_ms
    if use_tof:
        cdf_file['tof'] = delta_t_ps
    cdf_file['c1'] = c1
    cdf_file['c2'] = c2

    # Save as binary file
    # cdf_file_path = 'TB_J-PET_7th_gen_brain_insert_dz_1_mm_Comb' + tf_tag + '.cdf'
    cdf_file_path = 'Ermias_data' + tf_tag + '.cdf'
    binary_file = open(cdf_file_path, 'wb')
    binary_file.write(cdf_file.tobytes())
    binary_file.close()

    # Write the header file
    t_i = np.floor(np.min(time_ms) / 1000).astype(int)
    t_f = np.ceil(np.max(time_ms) / 1000).astype(int)
    tof_range = np.ceil((np.max(delta_t_ps) - np.min(delta_t_ps))).astype(int)
    castor_data_header_file(lut_path, cdf_file_path, num_entries, t_i, t_f, use_tof, tof_fwhm_ps, tof_range)

    return 0


def get_castor_id(gantry_id, rsector_id, crystal_id, layer_id):
    gantry_0_shape = np.array([2, 2, 24, 16 * 110], dtype=np.uint32)
    gantry_1_shape = np.array([2, 3, 24, 16 * 200], dtype=np.uint32)

    # Follow the CASToR hierarchy
    ring_idx = crystal_id
    rsector_idx = rsector_id

    if gantry_id == 0:
        layer_number = np.floor(layer_id / gantry_0_shape[3]).astype(np.uint32)
        layer_idx = layer_id - layer_number * gantry_0_shape[3]

        castor_id = np.ravel_multi_index((layer_number, ring_idx, rsector_idx, layer_idx), gantry_0_shape, order='C').astype(np.uint32)

    elif gantry_id == 1:
        layer_number = np.floor(layer_id / gantry_1_shape[3]).astype(np.uint32)
        layer_idx = layer_id - layer_number * gantry_1_shape[3]

        castor_id = np.ravel_multi_index((layer_number, ring_idx, rsector_idx, layer_idx), gantry_1_shape, order='C').astype(np.uint32) + np.prod(gantry_0_shape)

    else:
        sys.exit('Error')

    return castor_id


def get_castor_id2(gantry_id, rsector_id, crystal_id, layer_id, blur_z=True):
    # gantry_shape = np.array([[2, 2, 24, 16 * 110], [2, 3, 24, 16 * 200]], dtype=np.uint32)
    # gantry_shape = np.array([[2, 2, 24, 16 * 110], [2, 3, 24, 16 * 200], [2, 1, 10, 23 * 110]], dtype=np.uint32)
    # gantry_shape = np.array([[2, 2, 24, 16 * 110], [2, 3, 24, 16 * 200], [2, 1, 12, 16 * 110]], dtype=np.uint32)
    # gantry_shape = np.array([[2, 2, 24, 16, 110], [2, 3, 24, 16, 200], [2, 1, 12, 16, 110]], dtype=np.uint32)
    gantry_shape = np.array([[2, 2, 24, 16, 330], [2, 3, 24, 16, 600], [2, 1, 12, 16, 330]], dtype=np.uint32)
    # gantry_shape = np.array([[1, 1, 24, 13, 200]], dtype=np.uint32)
    z_fwhm = np.array([6., 6, 4.])
    shift = np.insert(gantry_shape.prod(axis=1).cumsum(), 0, 0)

    # print(shift)



    castor_id = np.zeros(gantry_id.shape, dtype=np.uint32)

    # layer_entries = gantry_shape[:, 3]
    layer_entries = gantry_shape[:, 3] * gantry_shape[:, 4]

    print(layer_entries)

    sys.exit()

    gantry_shape_red = gantry_shape[:, :4].copy()
    gantry_shape_red[:, -1] = layer_entries
    print(gantry_shape_red)

    for ii in range(gantry_shape.shape[0]):
        g = gantry_id == ii
        layer_number = np.floor(layer_id[g] / layer_entries[ii]).astype(np.uint32)
        layer_idx = layer_id[g] - layer_number * layer_entries[ii]

        print(np.unique(layer_idx))

        # Add more realistic uncertainty along z
        if blur_z:
            sc_idx, z_idx = np.unravel_index(layer_idx, gantry_shape[ii, 3:], order='F')
            z_idx_new = z_shuffling(z_idx, gantry_shape[ii, -1], z_fwhm[ii])
            layer_idx_new = np.ravel_multi_index((sc_idx, z_idx_new), gantry_shape[ii, 3:], order='F')
            layer_idx = layer_idx_new

        #layer_idx = np.ravel_multi_index((crystal_id[g], layer_idx), (13, 200), order='F')

        # castor_id[g] = np.ravel_multi_index((layer_number, crystal_id[g], rsector_id[g], layer_idx), gantry_shape[ii, :], order='C') + shift[ii]
        # castor_id[g] = np.ravel_multi_index((layer_number, crystal_id[g], rsector_id[g], layer_idx_new), gantry_shape_red[ii, :], order='C') + shift[ii]
        castor_id[g] = np.ravel_multi_index((layer_number, crystal_id[g], rsector_id[g], layer_idx), gantry_shape_red[ii, :], order='C') + shift[ii]
        # castor_id[g] = np.ravel_multi_index((layer_number, layer_number, rsector_id[g], layer_idx), gantry_shape_red[ii, :], order='C') + shift[ii]

    return castor_id


def z_shuffling(z_idx, n_z, z_fwhm):

    pmfs = truncated_gaussian_pmfs(n_z, dz=1., fwhm=z_fwhm, vis=False)
    z_idx_new = - np.ones(z_idx.shape, dtype=int)
    all_idx = np.arange(n_z, dtype=int)

    for ii in range(n_z):
        idx_temp = z_idx == ii
        temp = np.random.choice(all_idx, np.sum(idx_temp), replace=True, p=pmfs[ii, :])
        z_idx_new[idx_temp] = temp

    if np.sum(z_idx_new == -1) > 0:
        sys.exit('Error')

    return z_idx_new


def truncated_gaussian_pmfs(n, dz=1., fwhm=6., vis=False):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    mu = (np.arange(n)[:, np.newaxis] + 0.5) * dz
    x0 = np.arange(n)[np.newaxis, :] * dz
    x1 = (np.arange(n)[np.newaxis, :] + 1) * dz
    a = x0[0, 0]
    b = x1[0, -1]
    pmfs = (phi(x1, mu, sigma) - phi(x0, mu, sigma)) / (phi(b, mu, sigma) - phi(a, mu, sigma))

    # # Check the normalization
    # print(np.sum(pmfs, axis=1))

    if vis:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        im = ax.imshow(pmfs.T * 100, origin='lower', extent=[-0.5, n - 0.5, a, b])
        ax.set_xlabel('Index')
        ax.set_ylabel(r'$z$ [mm]')
        # ax.set_aspect('auto')
        ax.set_aspect(1)
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Probability [%]')
        # ax.set_xlim(right=15.5)
        # ax.set_ylim(top=15)
        plt.show()

    return pmfs


def phi(x, mu, sigma):
    return (1 + erf((x - mu) / (np.sqrt(2) * sigma))) / 2


if __name__ == "__main__":
    main()
