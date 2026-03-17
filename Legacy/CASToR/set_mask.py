"""
Set the mask image for CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np

# Auxiliary functions
from read_interfile import read_interfile
from write_interfile import write_interfile_binary
from vis import vis_3d


def main():
    x, y, z, img = read_interfile('/home/martin/J-PET/CASToR_RECONS/MASKS/mask.hdr', return_grid=True)
    # vis_3d(img)

    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing='ij')
    r_mask = 170  # [mm]

    img_mask = img.copy()
    img_mask[:] = 0.
    img_mask[x_mesh ** 2 + y_mesh ** 2 <= r_mask ** 2] = 1.

    # write_interfile_binary(img_mask, '/home/martin/J-PET/CASToR_RECONS/MASKS/', 'mask')



    return 0


if __name__ == "__main__":
    main()