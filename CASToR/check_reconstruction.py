"""


Author: Martin Rädler
"""
# Python libraries
import sys
from glob import glob

from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np

# Auxiliary functions
from read_interfile import read_interfile
from vis import vis_3d


def main():
    # img = read_interfile('/home/martin/J-PET/CASToR_RECONS/RECONS/Ermias/img_it2.hdr')
    img = read_interfile('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo/TB_J-PET_7th_gen_brain_insert_dz_1_mm_Comb/img_it100.hdr')
    # vis_3d(img)

    img_headers = natsorted(glob('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo/TB_J-PET_7th_gen_brain_insert_dz_1_mm_Comb/img_it*.hdr'))
    # img_headers = natsorted(glob('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo/TB_J-PET_7th_gen_brain_insert_dz_1_mm_Comb_2/img_it*.hdr'))

    x, y, z, _ = read_interfile(img_headers[0], return_grid=True)

    z_selection = np.abs(z - (-755.)) < 25

    img_tot = np.zeros((x.size, y.size, len(img_headers)))

    for ii in range(len(img_headers)):
        img_temp = read_interfile(img_headers[ii])
        img_tot[:, :, ii] = np.mean(img_temp[:, :, z_selection], axis=-1)


    vis_3d(img_tot, spacing=[1, 1, 1])

    return 0


if __name__ == "__main__":
    main()