"""
Axial correlation

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt

from CASToR.vis import vis_3d


def axial_correlation(img):

    img_centered = img - np.mean(img, axis=(0, 1))[np.newaxis, np.newaxis, :]
    # print(np.mean(img_centered, axis=(0, 1)))

    img_ref = img_centered[:, :, 25:26]

    aa = np.sum(img_ref * img_centered, axis=(0, 1)) / np.sqrt(np.sum(img_ref ** 2)) / np.sqrt(np.sum(img_centered ** 2, axis=(0, 1)))

    fig, ax = plt.subplots()
    ax.plot(aa)
    plt.show()
    print(aa.shape)

    # (np.sum(img_ref) / np.sqrt(np.sum(img_ref ** 2))) * img_centered / np.sqrt

    # vis_3d(img_centered)

    return 0


if __name__ == "__main__":
    pass
