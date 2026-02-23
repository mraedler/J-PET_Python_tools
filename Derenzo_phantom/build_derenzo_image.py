"""
Build the pixelated derenzo image
"""
# Python libraries
import sys
import numpy as np
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from tqdm import trange
from pickle import dump
import matplotlib.pyplot as plt

# Auxiliary functions
from build_derenzo_phantom import get_derenzo_parameters
from CASToR.read_interfile import read_interfile


def main():
    # Coordinate system from the reconstruction
    x, y, _, _ = read_interfile('/home/martin/J-PET/CASToR_RECONS/RECONS/Derenzo_400_ps_4_18_mm/2026-02-09_11-41-45_copy/ALL_true/img_it1.hdr', return_grid=True)

    x_grid = np.append(x - (x[1] - x[0]) / 2, 3 * x[-1] / 2 - x[-2] / 2)
    y_grid = np.append(y - (y[1] - y[0]) / 2, 3 * y[-1] / 2 - y[-2] / 2)

    boxes = polygonize_grid(x_grid, y_grid)
    boxes_tree = STRtree(boxes)

    # Get the Derenzo parameters
    x_peaks, y_peaks, r_peaks, x_valleys, y_valleys, r_valleys = get_derenzo_parameters(scaling_factor=3., return_valleys=True, visualize=False)
    radii = np.array([rr[0] for rr in r_peaks])

    # Construct the image
    img_peaks = np.zeros((x.size, y.size, len(x_peaks)))
    # img_valleys = np.zeros((x.size, y.size, len(x_peaks)))
    for ii in trange(0, len(x_peaks)):
        for jj in range(0, x_peaks[ii].size):
            img_peaks[:, :, ii] += pixelated_ellipse(boxes, boxes_tree, x_grid, y_grid, x_peaks[ii][jj], y_peaks[ii][jj], r_peaks[ii][jj], r_peaks[ii][jj], 0., vis=False)

        # for jj in range(0, x_valleys[ii].size):
        #     img_valleys[:, :, ii] += pixelated_ellipse(boxes, boxes_tree, x_grid, y_grid, x_peaks[ii][jj], y_peaks[ii][jj], r_peaks[ii][jj], r_peaks[ii][jj], 0., vis=False)

    fig, ax = plt.subplots()
    ax.imshow(np.sum(img_peaks, axis=-1).T, origin='lower')
    plt.show()


    np.save(sys.path[0] + '/Pixelated_ground_truth/x_peaks.npy', np.array(x_peaks, dtype=object), allow_pickle=True)
    np.save(sys.path[0] + '/Pixelated_ground_truth/y_peaks.npy', np.array(y_peaks, dtype=object), allow_pickle=True)
    # np.save(sys.path[0] + '/Pixelated_ground_truth/x_valleys.npy', np.array(x_valleys, dtype=object), allow_pickle=True)
    # np.save(sys.path[0] + '/Pixelated_ground_truth/y_valleys.npy', np.array(y_valleys, dtype=object), allow_pickle=True)

    np.save(sys.path[0] + '/Pixelated_ground_truth/x.npy', x)
    np.save(sys.path[0] + '/Pixelated_ground_truth/y.npy', y)
    np.save(sys.path[0] + '/Pixelated_ground_truth/x_grid.npy', x_grid)
    np.save(sys.path[0] + '/Pixelated_ground_truth/y_grid.npy', y_grid)
    np.save(sys.path[0] + '/Pixelated_ground_truth/radii.npy', radii)

    np.save(sys.path[0] + '/Pixelated_ground_truth/img_peaks.npy', img_peaks)
    # np.save(sys.path[0] + '/Pixelated_ground_truth/img_valleys.npy', img_valleys)
    # np.save(sys.path[0] + '/Pixelated_ground_truth/img_valleys_parzych.npy', img_valleys)

    return 0


def polygonize_grid(x_grid, y_grid):
    boxes = []
    for ii in range(x_grid.size - 1):
        for jj in range(y_grid.size - 1):
            g = box(x_grid[ii], y_grid[jj], x_grid[ii + 1], y_grid[jj + 1])
            boxes.append(g)

    return boxes


def pixelated_ellipse(boxes, boxes_tree, x_grid, y_grid, x_c, y_c, a, b, theta, vis=False):
    # Accuracy
    n_samples = 1000000
    t = np.arange(n_samples) / n_samples

    # Get the path and close it
    x_path, y_path = ellipse_parametrized(x_c, y_c, a, b, theta, t)
    x_path = np.append(x_path, x_path[0])
    y_path = np.append(y_path, y_path[0])

    # Define as shapely polygon
    poly = Polygon(np.column_stack([x_path, y_path]))
    assert poly.is_valid

    # Allocate
    img = np.zeros((x_grid.size - 1) * (y_grid.size - 1))

    # only pixels that intersect the polygon
    for kk in boxes_tree.query(poly, predicate="intersects"):
        intersection = poly.intersection(boxes[kk])
        if not intersection.is_empty:
            img[kk] = intersection.area
            # img[kk] = intersection.area / boxes[kk].area

    img = np.reshape(img, (x_grid.size - 1, y_grid.size - 1))

    area_regular_n_gon = n_samples / 2 * a * b * np.sin(2 * np.pi / n_samples)
    if np.abs(np.sum(img) - area_regular_n_gon) > 1e-12:
        print("Error above tolerance.")

    area_ellipse = np.pi * a * b
    print("Error with respect to the ellipse area: %1.2e." % np.abs(np.sum(img) - area_ellipse))

    # Normalize to one assuming uniform spacing
    img /= (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])

    if vis:
        fig, ax = plt.subplots()
        ax.imshow(img.T, origin='lower', extent=(x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
        ax.plot(x_path, y_path, color='tab:red')
        ax.set_xticks(x_grid)
        ax.set_yticks(y_grid)
        ax.grid()
        ax.set_xlim(x_c - a - b, x_c + a + b)
        ax.set_ylim(y_c - a - b, y_c + a + b)
        ax.set_aspect(1)
        plt.show()

    return img


def ellipse_parametrized(x_c, y_c, a, b, theta, t):
    x = a * np.cos(theta) * np.cos(2 * np.pi * t) - b * np.sin(theta) * np.sin(2 * np.pi * t) + x_c
    y = a * np.sin(theta) * np.cos(2 * np.pi * t) + b * np.cos(theta) * np.sin(2 * np.pi * t) + y_c
    return x, y


if __name__ == "__main__":
    main()
