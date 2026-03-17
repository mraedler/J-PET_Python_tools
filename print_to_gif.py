"""
Export a set of images as an animated gif
=========================================

Author: Martin Rädler
"""
# Libraries
import sys
import numpy as np
from imageio import get_writer
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def gif_img(img, file_name, window=None, step=1):
    # If no window is provided, use the minimum and maximum values of the center slice
    if window is None:
        center_slice = int(img.shape[-1] / 2)
        window = [np.min(img[:, :, center_slice]), np.max(img[:, :, center_slice])]

    # Map to [0, 1]
    img_mapped = (img - window[0]) / (window[1] - window[0])
    img_mapped[img_mapped < 0] = 0
    img_mapped[img_mapped > 1] = 1

    # Get the colormap
    viridis = get_cmap('viridis', 256)

    writer = get_writer(file_name, mode='I', fps=100, loop=True)

    for jj in tqdm(range(0, img.shape[-1], step)):
        # Apply the colormap
        img_temp = (viridis(img_mapped[:, :, jj])[:, :, :3] * 255).astype(np.uint8)

        # Add to gif
        writer.append_data(img_temp)

    writer.close()

    return 0


def gif_plot(img, file_name, extent=None, window=None, step=1):
    # If no extent is provided, assume unit pixel size
    if extent is None:
        extent = [0, img.shape[0], 0, img.shape[1]]

    # If no window is provided, use the minimum and maximum values of the center slice
    if window is None:
        center_slice = int(img.shape[-1] / 2)
        window = [np.min(img[:, :, center_slice]), np.max(img[:, :, center_slice])]

    fig, ax = plt.subplots()
    im = ax.imshow(img[:, :, 0].T, origin='lower', extent=extent, clim=window)
    tl = ax.set_title('Iteration number 0')

    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # writer = get_writer(file_name, mode='I', fps=10, loop=0)
    writer = get_writer(file_name, mode='I', fps=10)

    for jj in tqdm(range(0, img.shape[-1], step)):
        # Update the image
        im.set_data(img[:, :, jj].T)
        tl.set_text('Iteration number %d' % (jj + 1))

        # Get a screenshot of the plot
        fig.canvas.draw()
        screenshot = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        screenshot = screenshot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Add to the gif
        writer.append_data(screenshot)

    plt.close()
    writer.close()

    return 0
