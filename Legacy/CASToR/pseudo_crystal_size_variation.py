"""
Given the scintillator length and a target pseudo crystal size
Search for the number of pseudo crystals and their sizes that fit into the scintillator length

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt


def main():
    scintillator_length = 330.  # mm
    # delta_z_target = 4.  # mm
    delta_z_target = np.linspace(3, 10, 8)
    # delta_z_target = np.linspace(3, 10, 800)
    # delta_z_target = np.linspace(3, 15, 800)

    n_elements = scintillator_length / delta_z_target

    delta_z_above = scintillator_length / np.floor(n_elements)
    delta_z_below = scintillator_length / np.ceil(n_elements)
    delta_z_candidates = np.stack((delta_z_below, delta_z_above), axis=1)

    # difference = np.stack((delta_z_target - delta_z_below, delta_z_above - delta_z_target), axis=1)
    difference = np.abs(delta_z_candidates - delta_z_target[:, np.newaxis])


    idx = np.where(difference[:, 0] <= difference[:, 1], 0, 1)
    delta_z_opt = delta_z_candidates[np.arange(idx.size), idx]

    print(scintillator_length / delta_z_opt)


    # print(np.unique(delta_z_opt[(delta_z_opt - np.round(delta_z_opt, decimals=10)) == 0.]))

    delta_z_1 = np.array([3., 3.3, 3.4375, 3.75, 4.125, 4.4, 5., 5.15625, 5.5, 6., 6.6, 6.875, 7.5, 8.25, 10., 10.3125, 11., 13.2, 13.75, 15.])

    print(scintillator_length / delta_z_1)
    sys.exit()

    return 0


if __name__ == "__main__":
    main()
