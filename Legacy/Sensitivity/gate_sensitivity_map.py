"""
Sensitivity map for CASToR simulated by GATE

Author: Martin Rädler
"""
# Python libraries
import sys
from os import mkdir, makedirs
from os.path import isdir, isfile, basename, dirname, abspath
from glob import glob
from natsort import natsorted
import numpy as np
from uproot import open as open_root
from tqdm import trange

# Auxiliary functions
from data_structures import load_or_convert_to_structured_array
from CASToR.utilities import get_gantry_filter, get_true_filter
# from CASToR.vis import vis_3d


def main():
    script_dir = dirname(abspath(__file__))
    home_dir = dirname(dirname(script_dir))

    # Simulation data
    # geometry = '/SiPM_6mm_depth_3cm_cylinders'
    # geometry = '/SiPM_6mm_depth_3cm_box'
    # geometry = '/SiPM_6mm_depth_3cm_derenzo'
    # geometry = '/SiPM_6mm_depth_30mm_derenzo'
    # geometry = '/SiPM_4mm_depth_18mm_box'
    # geometry = '/TB_6_30_3_BI_4_18_3_Insert'
    geometry = '/TB_6_30_3_BI_4_18_3_Box'
    simulation_dirs = glob(home_dir + '/J-PET/Gate_Output/Sensitivity_maps' + geometry + '/*')
    simulation_dirs = natsorted(simulation_dirs)
    [print(sds) for sds in simulation_dirs]

    # Filter with respect to the interaction position
    gft = 'TOT'
    # gft = 'TBTB'
    # gft = 'TBB'
    # gft = 'BB'

    # Output directory
    output_dir = script_dir + '/Sensitivity_maps' + geometry + '/' + gft
    if not isdir(output_dir):
        # mkdir(output_dir)
        makedirs(output_dir)

    # Binning
    # x_edges, y_edges, z_edges, x, y, z = get_binning(n=[100+1, 100+1, 122+1], spacing=[10, 10, 20])
    # x_edges, y_edges, z_edges, x, y, z = get_binning(n=[202+1, 202+1, 122+1], spacing=[1, 1, 20])
    # x_edges, y_edges, z_edges, x, y, z = get_binning(n=[96 + 1, 96 + 1, 96 + 1], spacing=[5, 5, 5])
    x_edges, y_edges, z_edges, x, y, z = get_binning(n=[86 + 1, 86 + 1, 70 + 1], spacing=[5, 5, 5])
    z_edges, z = z_edges - 815., z - 815.
    save_if_changed(output_dir + '/x.npy', x)
    save_if_changed(output_dir + '/y.npy', y)
    save_if_changed(output_dir + '/z.npy', z)

    # Keys necessary to calculate the sensitivity map
    necessary_keys = ['eventID1', 'eventID2', 'comptonCrystal1', 'comptonCrystal2', 'gantryID1', 'gantryID2',
                      'rsectorID1', 'rsectorID2', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1']

    for ii in trange(len(simulation_dirs)):
        root_files = natsorted(glob(simulation_dirs[ii] + '/results_*.root'))
        for jj in trange(len(root_files)):
            # Load the simulation data
            root_file = open_root(root_files[jj])
            coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'], keys=necessary_keys, overwrite=False)

            # Data filters
            gantry_filter, gft = get_gantry_filter(coincidences_struct, gft)
            true_filter, tft = get_true_filter(coincidences_struct, filter_true=True)

            # Filter the data
            data = np.column_stack((coincidences_struct['sourcePosX1'], coincidences_struct['sourcePosY1'], coincidences_struct['sourcePosZ1']))
            data = data[true_filter & gantry_filter, :]

            # Calculate the histogram
            h, _ = np.histogramdd(data, bins=(x_edges, y_edges, z_edges))
            h = h.astype(int)
            # vis_3d(h)

            # Save
            np.save(output_dir + '/' + basename(simulation_dirs[ii]) + basename(root_files[jj])[7:-5] + '.npy', h)

            # Clear up memory
            del coincidences_struct, data, true_filter, gantry_filter

    # # For cylinders only
    # radii = np.array([410., 165., 410.])
    # half_z = np.array([932.5, 165., 117.5])
    # vol = np.sum(np.pi * radii ** 2 * half_z * 2)
    # print(vol / np.sum(true_coincidences))

    return 0


def get_binning(n, spacing):
    edges = []
    centers = []
    for ii in range(len(spacing)):
        edges_ii = symmetric_binning(n[ii]) * spacing[ii]
        edges.append(edges_ii)
        centers.append((edges_ii[1:] + edges_ii[:-1]) / 2)

    return *edges, *centers


def symmetric_binning(n):
    return np.arange(n) - (n - 1) / 2


def save_if_changed(path, new):
    changed = True

    if isfile(path):
        old = np.load(path)
        if new.size == old.size:
            if np.all(new == old):
                changed = False

    if changed:
        np.save(path, new)
        print('Updated: %s' % basename(path))

    return 0


if __name__ == "__main__":
    main()
