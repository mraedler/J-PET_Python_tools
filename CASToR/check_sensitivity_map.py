"""
Check the sensitivity map interpolated by CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

# Auxiliary functions
from read_interfile import read_interfile
from vis import vis_3d


def main():
    # sens = read_interfile('/home/martin/J-PET/CASToR_RECONS/SENS_MAPS/CONST_CASToR_ITP.hdr')
    sens = read_interfile('/home/martin/J-PET/CASToR_RECONS/SENS_MAPS/GATE_CASToR_ITP.hdr')
    mask = read_interfile('/home/martin/J-PET/CASToR_RECONS/MASKS/GATE_MASK_CASToR_ITP.hdr')

    # vis_3d(sens)
    # vis_3d(mask)

    print(np.min(sens[mask > 0]))

    fig, ax = plt.subplots()
    im = ax.imshow(mask[:, :, 0] > 0)
    plt.colorbar(im, ax=ax)
    plt.show()


    return 0


if __name__ == "__main__":
    main()
