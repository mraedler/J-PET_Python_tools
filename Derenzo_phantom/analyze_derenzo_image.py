"""

"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

# Auxiliary functions
from CASToR.read_interfile import read_interfile
from CASToR.vis import vis_3d
from get_derenzo_image import get_ground_truth_derenzo_image, load_derenzo_image
from get_derenzo_contrast import get_derenzo_contrast_function


def main():
    # iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm/2026-02-11_15-37-48/TB-BI_true/')
    iterations, imgs = load_derenzo_image('/Derenzo_400_ps_4_18_mm_outside_non_collinearity/2026-02-21_20-22-47/TB-TB_true/', z_offset=755)

    # print(imgs.shape)
    vis_3d(imgs[:, :, :, -1])

    return 0


if __name__ == "__main__":
    main()
