"""
Phantom

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt

# Auxiliary functions
from pixelate_contour import t_parametrization, pixelate_contour


def main():
    x_grid = np.arange(-200, 200 + 1)
    y_grid = np.arange(-200, 200 + 1)

    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    x_rods = np.concatenate(np.load(sys.path[0] + '/Hexagon/x_rods.npy', allow_pickle=True))
    y_rods = np.concatenate(np.load(sys.path[0] + '/Hexagon/y_rods.npy', allow_pickle=True))
    radii = np.load(sys.path[0] + '/Hexagon/radii.npy') * 2

    # Construct the images
    # img_hexagon = np.zeros((x.size, y.size, x_rods.size))
    img_hexagon = np.zeros((x.size, y.size, x_rods.size), dtype=bool)

    n_contour = 1000
    t = np.arange(n_contour) / n_contour * 2 * np.pi

    for ii in trange(x_rods.size):
        x_contour, y_contour = t_parametrization(x_rods[ii], y_rods[ii], radii, radii, 0., t)
        img_rod = pixelate_contour(x_grid, y_grid, x_contour, y_contour, show_figure=True)
        error = np.pi * radii ** 2 - np.sum(img_rod)
        if np.abs(error) > 4e-4:
            print('Error too large!')
        img_hexagon[:, :, ii] = img_rod
        # img_hexagon[:, :, ii] = img_rod > 0.5

    # np.save(sys.path[0] + '/Hexagon/img_hexagon.npy', img_hexagon)
    # np.save(sys.path[0] + '/Hexagon/img_mask.npy', img_hexagon)

    fig, ax = plt.subplots()
    ax.imshow(np.sum(img_hexagon, axis=-1).T, origin='lower', extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
    plt.show()

    return 0


if __name__ == "__main__":
    main()
