"""
Accumulate the sensitivity maps from separate GATE simulations

Author: Martin Rädler
"""
# Python libraries
import sys
from glob import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Auxiliary functions
from vis import vis_3d
from utilities import get_extent
from write_interfile import write_sensitivity_map
from read_interfile import read_interfile


def main():
    gft = 'TOT'
    # gft = 'TBTB'
    # gft = 'TBB'
    # gft = 'BB'

    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_cylinders/TOT'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_cylinders/TB-TB'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_cylinders/B-B'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_cylinders/TB-B'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_derenzo/tot'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_30mm_derenzo/tot'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_4mm_depth_18mm_box/bb'
    input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/TB_6_30_3_BI_4_18_3_Insert'
    # input_dir = '/home/martin/PycharmProjects/J-PET/Sensitivity_maps/TB_6_30_3_BI_4_18_3_Box'

    input_dir += '/' + gft

    x = np.load(input_dir + '/x.npy')
    y = np.load(input_dir + '/y.npy')
    z = np.load(input_dir + '/z.npy')

    files = glob(input_dir + '/2*.npy')
    print(len(files))

    acc = np.zeros((x.size, y.size, z.size), dtype=np.int64)
    for ii in range(len(files)):
        acc += np.load(files[ii])

    # acc[acc == 0] = 1
    # acc[acc > 0] = 1

    mid = (np.array(acc.shape) / 2).astype(int)
    mid_acc = acc[mid[0], mid[1], mid[2]]
    print(mid_acc)

    # print(np.sum(acc) / acc.size)

    acc = acc.astype(np.float32)

    # show_sensitivity_map(x, y, z, acc)
    vis_3d(acc, spacing=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], axis=-1)
    vis_3d(acc, spacing=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], transpose=True, axis=-3)
    # np.save(input_dir + '/Accumulated.npy', acc)

    # acc_ref = np.load('/home/martin/PycharmProjects/J-PET/Sensitivity_maps/SiPM_6mm_depth_3cm_cylinders/TOT/Accumulated.npy')
    # nn_min = 48
    # nn_max = 51
    # down_scaling_factor = (np.sum(acc[nn_min:nn_max+1, nn_min:nn_max+1, z > 0] * acc_ref[nn_min:nn_max+1, nn_min:nn_max+1, z > 0]) /
    #                        np.sum(acc[nn_min:nn_max+1, nn_min:nn_max+1, z > 0] ** 2))
    # down_scaling_factor = 0.039599877175500384

    # acc = np.ones(acc.shape)
    # acc[:, :, 61:] = 0
    # acc[:, :, :62] = 0
    # acc[acc == 0] = 1e10
    # acc = np.flip(acc, axis=2)

    # vis_3d(acc, spacing=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], transpose=True, axis=-3)
    # sys.exit()



    # acc *= down_scaling_factor

    # fig, ax = plt.subplots()
    # nn = 50
    # ax.plot(z, acc_ref[nn, nn, :])
    # ax.plot(z, acc[nn, nn, :])
    # plt.show()


    # """Resample"""
    # x_new, y_new, z_new, _ = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/sensitivity_maps/TB_brain_CASToR.hdr', return_grid=True)
    # acc_new = resample_volume(x, y, z, acc, x_new, y_new, z_new)
    # x, y, z, acc = x_new.copy(), y_new.copy(), z_new.copy(), acc_new.copy()

    """Export as interfile"""
    # name = 'TB_brain_2_bb_GATE'
    # scanner = 'TB_JPET_6th_gen_7_rings_gap_2cm_Brain_2'
    # name = 'TB_brain_tot_GATE'
    # scanner = 'TB_JPET_6th_gen_7_rings_gap_2cm_Brain'

    # name = gft
    scanner = 'TB_6_30_3_BI_4_18_3'

    output_path = '/home/martin/J-PET/CASToR_RECONS/SENS_MAPS/GATE/' + scanner + '/'

    write_sensitivity_map(x, y, z, acc, output_path, gft, scanner)

    return 0


def resample_volume(x, y, z, vol, x_new, y_new, z_new):
    interp = RegularGridInterpolator((x, y, z), vol, bounds_error=False, fill_value=0)
    vol_new = interp(tuple(np.meshgrid(x_new, y_new, z_new, indexing='ij')))
    # vis_3d(vol_new, transpose=True, axis=0)
    return vol_new


def show_sensitivity_map(x, y, z, vol):

    vol = vol.astype(np.float32)
    vol /= np.max(vol[50, :, :]) / 100


    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(1, 2, width_ratios=(1, 2.55), figsize=(14, 4))

    ax0.imshow(vol[:, :, 61], extent=get_extent(x, y), clim=(0, 100))
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')

    im = ax1.imshow(vol[50, :, :], extent=get_extent(z, x), clim=(0, 100))
    ax1.set_xlabel(r'$z$ [mm]')

    cax = make_axes_locatable(ax1).append_axes('right', size='3%', pad=0.1)

    c_bar = fig.colorbar(im, cax=cax, orientation='vertical')
    c_bar.set_ticks([0, 20, 40, 60, 80, 100])
    c_bar.set_label('Sensitivity [%]')
    print(dir(c_bar))

    plt.show()

    return 0


if __name__ == "__main__":
    main()
