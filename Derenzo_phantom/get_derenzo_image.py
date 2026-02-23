"""
Convenience function to load the pixelated derenzo image

Author: Martin Rädler
"""
# Python libraries
import sys
import os
from glob import glob
from pathlib import Path
from natsort import natsorted
import numpy as np

# Auxiliary functions
from CASToR.read_interfile import read_interfile


def get_ground_truth_derenzo_image():

    # x = np.load(sys.path[0] + "/Pixelated_ground_truth/x.npy")
    # y = np.load(sys.path[0] + "/Pixelated_ground_truth/y.npy")
    #
    # x_grid = np.load(sys.path[0] + "/Pixelated_ground_truth/x_grid.npy")
    # y_grid = np.load(sys.path[0] + "/Pixelated_ground_truth/y_grid.npy")
    #
    # x_peaks = np.load(sys.path[0] + "/Pixelated_ground_truth/x_peaks.npy", allow_pickle=True)
    # y_peaks = np.load(sys.path[0] + "/Pixelated_ground_truth/y_peaks.npy", allow_pickle=True)
    #
    # radii = np.load(sys.path[0] + "/Pixelated_ground_truth/radii.npy")
    #
    # img_peaks = np.load(sys.path[0] + "/Pixelated_ground_truth/img_peaks.npy")

    path = "/home/martin/PycharmProjects/J-PET_Python_tools/Derenzo_phantom"

    x = np.load(path + "/Pixelated_ground_truth/x.npy")
    y = np.load(path + "/Pixelated_ground_truth/y.npy")

    x_grid = np.load(path + "/Pixelated_ground_truth/x_grid.npy")
    y_grid = np.load(path + "/Pixelated_ground_truth/y_grid.npy")

    x_peaks = np.load(path + "/Pixelated_ground_truth/x_peaks.npy", allow_pickle=True)
    y_peaks = np.load(path + "/Pixelated_ground_truth/y_peaks.npy", allow_pickle=True)

    radii = np.load(path + "/Pixelated_ground_truth/radii.npy")

    img_peaks = np.load(path + "/Pixelated_ground_truth/img_peaks.npy")

    return x, y, x_grid, y_grid, x_peaks, y_peaks, radii, img_peaks


def load_derenzo_image(reconstruction_dir, it=False, z_offset=-755):
    castor_output_dir = '/home/martin/J-PET/CASToR_RECONS/RECONS'
    iterations = np.concatenate([np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 1000, 100), np.array([1000])])

    aa = natsorted(glob(castor_output_dir + reconstruction_dir + '/img_it*.hdr'))
    iterations_available = np.array([int(Path(entry).stem[6:]) for entry in aa])
    print(iterations_available)
    iterations = iterations_available

    x, y, z, img_1 = read_interfile(castor_output_dir + reconstruction_dir + '/img_it1.hdr', return_grid=True)
    z_selection = np.abs(z - z_offset) < 25.

    if it:
        idx = np.argwhere(iterations == it)
        if idx.size == 1:
            img = read_interfile(castor_output_dir + reconstruction_dir + '/img_it%d.hdr' % iterations[idx])
            return img[:, :, z_selection]
        else:
            sys.exit("Iteration number %d not available." % it)

    imgs = np.zeros((x.size, y.size, np.sum(z_selection), iterations.size))

    for ii in range(iterations.size):
        img = read_interfile(castor_output_dir + reconstruction_dir + '/img_it%d.hdr' % iterations[ii])
        imgs[:, :, :, ii] = img[:, :, z_selection]

    return iterations, imgs


if __name__ == "__main__":
    get_ground_truth_derenzo_image()
