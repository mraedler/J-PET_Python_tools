"""

"""
# Python libraries
import sys
import numpy as np

# Auxiliary functions
from CASToR.read_interfile import read_interfile
from CASToR.vis import vis_3d


def main():
    iterations = np.concatenate([np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 700, 100)])
    # iterations = np.array([1])

    x, y, z, first_iteration = read_interfile('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo_400_ps_4_18_mm/2026-02-09_11-41-45_copy/TB-TB_true/img_it1.hdr', return_grid=True)
    z_selection = np.abs(z - (-755.)) < 25.

    collected = np.zeros((first_iteration.shape[0], first_iteration.shape[1], iterations.size))
    for ii in range(iterations.size):
        img = read_interfile('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo_400_ps_4_18_mm/2026-02-09_11-41-45_copy/ALL_true/img_it%d.hdr' % iterations[ii])
        # vis_3d(img[:, :, z_selection])
        collected[:, :, ii] = np.mean(img[:, :, z_selection], axis=-1)

    vis_3d(collected)

    return 0


if __name__ == "__main__":
    main()
