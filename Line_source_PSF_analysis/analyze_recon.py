"""
Analyze reconstructions from CASToR

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from uproot import open as open_root
from scipy.interpolate import interp1d
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Auxiliary functions
from read_interfile import read_interfile
from utilities import get_extent
sys.path.append('../')
from sensitivity_coincidences import get_sensitivity, plot_sensitivity
from layer_utilization import equate_layer_utilization
from data_structures import load_or_convert_to_structured_array
from vis import vis_3d


def main():
    # Binning for the line sensitivity
    n_bins = 80
    z_edges = np.linspace(-1200., 1200., n_bins + 1)
    z_centers = (z_edges[1:] + z_edges[:-1]) / 2
    z_widths = z_edges[1:] - z_edges[:-1]

    """TB only"""
    # Reconstructions
    _, _, z, tb_all_castor = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_only_all_CASToR_it4.hdr', return_grid=True)
    tb_true_castor = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_only_true_CASToR_it4.hdr')
    tb_all_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_only_all_CONST_it4.hdr')
    tb_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_only_true_CONST_it4.hdr')

    # Line source sensitivity
    root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-04-11_10-03-38/results.root')  # TB only
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'])
    # coincidences_struct = equate_layer_utilization(coincidences_struct)
    h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain = get_sensitivity(coincidences_struct, z_edges)

    show_plot = False
    if show_plot:
        # Plot
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        # ax.plot(z, tb_all_castor[10, 10, :], label='CASToR sens. (all)')
        # ax.plot(z, tb_true_castor[10, 10, :], label='CASToR sens. (true)')
        # ax.plot(z, tb_all_const[10, 10, :], label='CONST sens. (all)')
        # ax.plot(z, tb_true_const[10, 10, :], label='CONST sens. (true)')

        ax.plot(z, np.mean(tb_all_const, axis=(0, 1)), label='CONST sens. (all)')
        ax.plot(z_centers, h_raw / 1e10, label='GATE sens. (all)')
        ax.plot(z, np.mean(tb_true_const, axis=(0, 1)), label='CONST sens. (true)')
        ax.plot(z_centers, h_filtered / 1e10, label='GATE sens. (true)')

        ax.set_xlabel(r'$z$ [mm]')
        ax.legend()
        plt.show()

    """TB brain"""
    # Reconstructions
    # x, y, z, tbb_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_true_CONST_it4.hdr', return_grid=True)
    # tbb_all_castor = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_brain_all_CASToR_it4.hdr')
    # tbb_true_gate_itp = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/img_TB_brain_true_GATE_ITP_it4.hdr')

    # img_tot_all_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_all_GATE_it4.hdr')
    # img_tot_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_true_GATE_it4.hdr')
    # img_tbtb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_true_tbtb_GATE_it4.hdr')
    # img_bb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_true_bb_GATE_it4.hdr')
    # img_tbb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_10s/img_TB_brain_true_tbb_GATE_it4.hdr')

    # 100s
    x, y, z, img_tot_all_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tot_all_GATE_it4.hdr', return_grid=True)
    img_tot_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tot_true_GATE_it4.hdr')
    img_tbtb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tbtb_true_GATE_it4.hdr')
    img_bb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_bb_true_GATE_it4.hdr')
    img_tbb_true_gate = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tbb_true_GATE_it4.hdr')
    img_tot_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tot_true_CONST_it4.hdr')
    img_tbtb_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tbtb_true_CONST_it4.hdr')
    img_bb_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_bb_true_CONST_it4.hdr')
    img_tbb_true_const = read_interfile('/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s/img_TB_brain_tbb_true_CONST_it4.hdr')

    # Line source sensitivity
    # root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-03-12_15-12-50/results.root')  # TB-B: 10 s
    root_file = open_root('/home/martin/J-PET/Gate_mac_9.3/TB_J-PET_Brain_9.3/Output/Line_source/2024-05-28_15-27-04/results.root')  # TB-B: 100 s
    coincidences_struct = load_or_convert_to_structured_array(root_file['MergedCoincidences'])
    # coincidences_struct = equate_layer_utilization(coincidences_struct)
    h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain = get_sensitivity(coincidences_struct, z_edges)
    plot_sensitivity(z_edges, z_centers, z_widths, h_raw, h_filtered, h_filtered_total_body, h_filtered_separate, h_filtered_brain, 1000, [-815 - 330 / 2, -815 + 330 / 2])
    sys.exit()

    # Plot
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, img_tot_all_gate[10, 10, :], label=r'$f(0,0,z)$')
    ax.plot(z, img_tot_true_gate[10, 10, :], label=r'$f(0,0,z)$ (true only)')
    ax.plot(z, np.sum(img_tot_all_gate, axis=(0, 1)) / 1e1, label=r'$\int f(x,y,z)\mathrm{d}x\mathrm{d}y$')
    ax.plot(z, np.sum(img_tot_true_gate, axis=(0, 1)) / 1e1, label=r'$\int f(x,y,z)\mathrm{d}x\mathrm{d}y$ (true only)')
    ax.set_ylim(0, 1e-2)
    ax.set_title('Using the GATE sensitivity map')

    # ax.plot(z, np.sum(img_tot_true_const, axis=(0, 1)), label=r'$\int f(x,y,z)\mathrm{d}x\mathrm{d}y$')
    # ax.plot(z_centers, h_true / 3e1, label='Sensitivity profile')
    # ax.set_title('Using no/constant sensitivity map')

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel(r'$z$ [mm]')
    ax.legend(loc='upper center', ncol=2, frameon=False)
    plt.show()

    # radial_profile(x, y, z, img_tot_true_gate)
    # radial_profile(x, y, z, img_tot_all_gate)

    """Profile fits"""
    # z_values = np.array([-800, 800], ndmin=2)  # mm
    # z_indices = np.argmin(np.abs(z[:, np.newaxis] - z_values), axis=0)
    # slices = img_tot_true_gate[:, :, z_indices]
    # # profile_fit_comparison(x, y, slices[:, :, 1])
    #
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    # ax0.imshow(slices[:, :, 0], clim=(0, np.max(slices[:, :, 1])))
    # ax0.set_title(r'$z=-800$ mm')
    # im = ax1.imshow(slices[:, :, 1], clim=(0, np.max(slices[:, :, 1])))
    # ax1.set_title(r'$z=800$ mm')
    # cb = plt.colorbar(im, ax=ax1)
    # cb.formatter.set_powerlimits((0, 0))
    # plt.show()

    return 0


def radial_profile(x, y, z, img):
    #
    interp = RegularGridInterpolator((x, y, z), img, bounds_error=False, fill_value=0)
    radii = np.arange(9 + 1)  # [mm]
    profiles = np.zeros((radii.size, z.size))
    for ii in tqdm(range(radii.size)):
        xs, ys = get_radial_sampling_points(radii[ii], 1, visualize=False)

        xs_mesh, z_mesh = np.meshgrid(xs, z, indexing='ij')
        ys_mesh, _ = np.meshgrid(ys, z, indexing='ij')
        img_itp = interp((xs_mesh, ys_mesh, z_mesh))
        profiles[ii, :] = np.mean(img_itp, axis=0)

        # img_itp = interp(tuple(np.meshgrid(xs, ys, z, indexing='ij')))
        # profiles[ii, :] = np.mean(img_itp, axis=(0, 1))

    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # extent = [x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
    #
    # fig, ax = plt.subplots()
    # ax.imshow(img[:, :, 1215], extent=extent)
    # plt.show()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plateau_domain = z > 0
    quadratic_domain = (z > -1000) & (z < -250)

    fig, ax = plt.subplots()
    for ii in range(profiles.shape[0]):
        plateau = np.polyfit(z[plateau_domain], profiles[ii, plateau_domain], 0)[0]
        quadratic = np.polyfit(z[quadratic_domain], profiles[ii, quadratic_domain], 2)

        z_min = - quadratic[1] / (2 * quadratic[0])
        p_min = quadratic[2] - quadratic[1] ** 2 / (4 * quadratic[0])

        print('%1.1f %%' % ((plateau - p_min) / plateau * 100))

        ax.plot(z, profiles[ii, :], color=colors[ii], alpha=0.75)
        ax.plot([z[0], z[-1]], [plateau, plateau], linestyle='--', color=colors[ii])
        ax.plot(z[quadratic_domain], quadratic[0] * z[quadratic_domain] ** 2 + quadratic[1] * z[quadratic_domain] + quadratic[2], linestyle=':', color=colors[ii])
        ax.plot(z_min, p_min, 'o')

    plt.show()

    return 0


def get_radial_sampling_points(r, r_delta_phi, visualize=False):
    r_2_pi = r * 2 * np.pi
    n = np.ceil(r_2_pi / r_delta_phi).astype(int)
    if n == 0:
        n = 1
    phis = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    xs, ys = r * np.cos(phis), r * np.sin(phis)

    if visualize:
        fig, ax = plt.subplots()
        ax.scatter(xs, ys)
        ax.add_artist(Circle((0, 0), r, facecolor='none', edgecolor='black', linestyle='--'))
        ax.set_xlim(-r-.1, r+.1)
        ax.set_ylim(-r-.1, r+.1)
        ax.set_aspect('equal')
        plt.show()

    return xs, ys


if __name__ == "__main__":
    main()
