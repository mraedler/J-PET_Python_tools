"""
Estimate the FWHMs via fits

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Auxiliary functions
from CASToR.utilities import get_extent
from CASToR.read_interfile import read_interfile
from CASToR.vis import vis_3d


def main():
    # check_distribution_normalization(show_distribution=True)
    # fwhm_fits_sipm_6mm_depth_30mm()
    fwhm_fits_sipm_4mm_depth_18mm()

    return 0


def fwhm_fits_sipm_6mm_depth_30mm():
    # Load the images
    interfile_path = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s'
    x, y, z, img_tot_all_gate = read_interfile(interfile_path + '/img_TB_brain_tot_all_GATE_it4.hdr', return_grid=True)
    img_tot_true_gate = read_interfile(interfile_path + '/img_TB_brain_tot_true_GATE_it4.hdr')
    img_tbtb_true_gate = read_interfile(interfile_path + '/img_TB_brain_tbtb_true_GATE_it4.hdr')
    img_bb_true_gate = read_interfile(interfile_path + '/img_TB_brain_bb_true_GATE_it4.hdr')
    img_tbb_true_gate = read_interfile(interfile_path + '/img_TB_brain_tbb_true_GATE_it4.hdr')
    img_tot_true_const = read_interfile(interfile_path + '/img_TB_brain_tot_true_CONST_it4.hdr')
    img_tbtb_true_const = read_interfile(interfile_path + '/img_TB_brain_tbtb_true_CONST_it4.hdr')
    img_bb_true_const = read_interfile(interfile_path + '/img_TB_brain_bb_true_CONST_it4.hdr')
    img_tbb_true_const = read_interfile(interfile_path + '/img_TB_brain_tbb_true_CONST_it4.hdr')

    # FWHM dir
    fwhm_path = '/home/martin/PycharmProjects/J-PET/FWHMs/SiPM_6mm_depth_30mm_line_source'
    # np.save(fwhm_path + '/z.npy', z)
    # np.save(fwhm_path + '/w_tot_true.npy', np.sum(img_tot_true_const, axis=(0, 1)))
    # np.save(fwhm_path + '/w_tbtb_true.npy', np.sum(img_tbtb_true_const, axis=(0, 1)))
    # np.save(fwhm_path + '/w_bb_true.npy', np.sum(img_bb_true_const, axis=(0, 1)))
    # np.save(fwhm_path + '/w_tbb_true.npy', np.sum(img_tbb_true_const, axis=(0, 1)))

    # Estimate FWHM and save
    fwhm_tot_all_gauss, _, me_tot_all_gauss = profile_fits(x, y, img_tot_all_gate, fit_function='gaussian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    # fwhm_tot_true_gauss, _, me_tot_true_gauss = profile_fits(x, y, img_tot_true_gate, fit_function='gaussian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    # fwhm_tbtb_true_gauss, _, me_tbtb_true_gauss = profile_fits(x, y, img_tbtb_true_gate, fit_function='gaussian', z_include=np.sum(img_tbtb_true_const, axis=(0, 1))>.1)
    # fwhm_bb_true_gauss, _, me_bb_true_gauss = profile_fits(x, y, img_bb_true_gate, fit_function='gaussian', z_include=np.sum(img_bb_true_const, axis=(0, 1))>.1)
    # fwhm_tbb_true_gauss, _, me_tbb_true_gauss = profile_fits(x, y, img_tbb_true_gate, fit_function='gaussian', z_include=np.sum(img_tbb_true_const, axis=(0, 1))>.1)

    # np.save(fwhm_path + '/fwhm_tot_all_gauss.npy', fwhm_tot_all_gauss)
    # np.save(fwhm_path + '/fwhm_tot_true_gauss.npy', fwhm_tot_true_gauss)
    # np.save(fwhm_path + '/fwhm_tbtb_true_gauss.npy', fwhm_tbtb_true_gauss)
    # np.save(fwhm_path + '/fwhm_bb_true_gauss.npy', fwhm_bb_true_gauss)
    # np.save(fwhm_path + '/fwhm_tbb_true_gauss.npy', fwhm_tbb_true_gauss)

    fwhm_tot_all_lorentz, _, me_tot_all_lorentz = profile_fits(x, y, img_tot_all_gate, fit_function='lorentzian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    # fwhm_tot_true_lorentz, _, me_tot_true_lorentz = profile_fits(x, y, img_tot_true_gate, fit_function='lorentzian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    # fwhm_tbtb_true_lorentz, _, me_tbtb_true_lorentz = profile_fits(x, y, img_tbtb_true_gate, fit_function='lorentzian', z_include=np.sum(img_tbtb_true_const, axis=(0, 1))>.1)
    # fwhm_bb_true_lorentz, _, me_bb_true_lorentz = profile_fits(x, y, img_bb_true_gate, fit_function='lorentzian', z_include=np.sum(img_bb_true_const, axis=(0, 1))>.1)
    # fwhm_tbb_true_lorentz, _, me_tbb_true_lorentz = profile_fits(x, y, img_tbb_true_gate, fit_function='lorentzian', z_include=np.sum(img_tbb_true_const, axis=(0, 1))>.1)
    #
    # np.save(fwhm_path + '/fwhm_tot_all_lorentz.npy', fwhm_tot_all_lorentz)
    # np.save(fwhm_path + '/fwhm_tot_true_lorentz.npy', fwhm_tot_true_lorentz)
    # np.save(fwhm_path + '/fwhm_tbtb_true_lorentz.npy', fwhm_tbtb_true_lorentz)
    # np.save(fwhm_path + '/fwhm_bb_true_lorentz.npy', fwhm_bb_true_lorentz)
    # np.save(fwhm_path + '/fwhm_tbb_true_lorentz.npy', fwhm_tbb_true_lorentz)
    sys.exit()

    fwhm_tot_all_genlorentz, _, me_tot_all_lorentz = profile_fits(x, y, img_tot_all_gate, fit_function='generalized_lorentzian')
    fwhm_tot_true_genlorentz, _, me_tot_true_lorentz = profile_fits(x, y, img_tot_true_gate, fit_function='generalized_lorentzian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    fwhm_tbtb_true_genlorentz, _, me_tbtb_true_lorentz = profile_fits(x, y, img_tbtb_true_gate, fit_function='generalized_lorentzian', z_include=np.sum(img_tbtb_true_const, axis=(0, 1))>.1)
    fwhm_bb_true_genlorentz, _, me_bb_true_lorentz = profile_fits(x, y, img_bb_true_gate, fit_function='generalized_lorentzian', z_include=np.sum(img_bb_true_const, axis=(0, 1))>.1)
    fwhm_tbb_true_genlorentz, _, me_tbb_true_lorentz = profile_fits(x, y, img_tbb_true_gate, fit_function='generalized_lorentzian', z_include=np.sum(img_tbb_true_const, axis=(0, 1))>.1)

    np.save(fwhm_path + '/fwhm_tot_all_genlorentz.npy', fwhm_tot_all_genlorentz)
    np.save(fwhm_path + '/fwhm_tot_true_genlorentz.npy', fwhm_tot_true_genlorentz)
    np.save(fwhm_path + '/fwhm_tbtb_true_genlorentz.npy', fwhm_tbtb_true_genlorentz)
    np.save(fwhm_path + '/fwhm_bb_true_genlorentz.npy', fwhm_bb_true_genlorentz)
    np.save(fwhm_path + '/fwhm_tbb_true_genlorentz.npy', fwhm_tbb_true_genlorentz)

    # compare_fit_error(z, [me_tot_all_gauss], [me_tot_all_lorentz], ['TOT'])
    compare_fit_error(z, [me_tot_true_gauss, me_tbtb_true_gauss, me_bb_true_gauss, me_tbb_true_gauss],
                      [me_tot_true_lorentz, me_tbtb_true_lorentz, me_bb_true_lorentz, me_tbb_true_lorentz], ['TOT', 'TB-TB', 'B-B', 'TB-B'])
    return 0


def fwhm_fits_sipm_4mm_depth_18mm():
    # Load the images
    interfile_path = '/home/martin/J-PET/CASToR_scripts/recon/reconstructed_images/line_source_100s_2'
    x, y, z, img_tot_all_gate = read_interfile(interfile_path + '/img_TB_brain_2_tot_all_GATE_it4.hdr', return_grid=True)
    img_tot_true_gate = read_interfile(interfile_path + '/img_TB_brain_2_tot_true_GATE_it4.hdr')
    img_tbtb_true_gate = read_interfile(interfile_path + '/img_TB_brain_2_tbtb_true_GATE_it4.hdr')
    img_tbb_true_gate = read_interfile(interfile_path + '/img_TB_brain_2_tbb_true_GATE_it4.hdr')
    img_bb_true_gate = read_interfile(interfile_path + '/img_TB_brain_2_bb_true_GATE_it4.hdr')

    img_tot_true_const = read_interfile(interfile_path + '/img_TB_brain_2_tot_true_CONST_it4.hdr')
    img_tbtb_true_const = read_interfile(interfile_path + '/img_TB_brain_2_tbtb_true_CONST_it4.hdr')
    img_tbb_true_const = read_interfile(interfile_path + '/img_TB_brain_2_tbb_true_CONST_it4.hdr')
    img_bb_true_const = read_interfile(interfile_path + '/img_TB_brain_2_bb_true_CONST_it4.hdr')

    # Fit the profiles
    fwhm_tot_all_lorentz, _, me_tot_true_lorentz = profile_fits(x, y, img_tot_all_gate, fit_function='lorentzian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    fwhm_tot_true_lorentz, _, me_tot_true_lorentz = profile_fits(x, y, img_tot_true_gate, fit_function='lorentzian', z_include=np.sum(img_tot_true_const, axis=(0, 1))>.1)
    fwhm_tbtb_true_lorentz, _, me_tbtb_true_lorentz = profile_fits(x, y, img_tbtb_true_gate, fit_function='lorentzian', z_include=np.sum(img_tbtb_true_const, axis=(0, 1))>.1)
    fwhm_bb_true_lorentz, _, me_bb_true_lorentz = profile_fits(x, y, img_bb_true_gate, fit_function='lorentzian', z_include=np.sum(img_bb_true_const, axis=(0, 1))>.1)
    fwhm_tbb_true_lorentz, _, me_tbb_true_lorentz = profile_fits(x, y, img_tbb_true_gate, fit_function='lorentzian', z_include=np.sum(img_tbb_true_const, axis=(0, 1))>.1)

    # fig, ax = plt.subplots()
    # ax.plot(z, me_tot_true_lorentz)
    # ax.plot(z, me_tbtb_true_lorentz)
    # ax.plot(z, me_bb_true_lorentz)
    # ax.plot(z, me_tbb_true_lorentz)
    # plt.show()

    # Save the results
    fwhm_path = '/home/martin/PycharmProjects/J-PET/FWHMs/SiPM_4mm_depth_18mm_line_source'
    np.save(fwhm_path + '/z.npy', z)
    np.save(fwhm_path + '/w_tot_true.npy', np.sum(img_tot_true_const, axis=(0, 1)))
    np.save(fwhm_path + '/w_tbtb_true.npy', np.sum(img_tbtb_true_const, axis=(0, 1)))
    np.save(fwhm_path + '/w_bb_true.npy', np.sum(img_bb_true_const, axis=(0, 1)))
    np.save(fwhm_path + '/w_tbb_true.npy', np.sum(img_tbb_true_const, axis=(0, 1)))

    np.save(fwhm_path + '/z.npy', z)
    np.save(fwhm_path + '/fwhm_tot_all_lorentz.npy', fwhm_tot_all_lorentz)
    np.save(fwhm_path + '/fwhm_tot_true_lorentz.npy', fwhm_tot_true_lorentz)
    np.save(fwhm_path + '/fwhm_tbtb_true_lorentz.npy', fwhm_tbtb_true_lorentz)
    np.save(fwhm_path + '/fwhm_bb_true_lorentz.npy', fwhm_bb_true_lorentz)
    np.save(fwhm_path + '/fwhm_tbb_true_lorentz.npy', fwhm_tbb_true_lorentz)

    return 0


# noinspection PyTupleAssignmentBalance
def profile_fits(x, y, slices, fit_function='lorentzian', z_include=True):
    # Fit the models to the data
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')

    # Allocate
    fwhm = np.zeros(slices.shape[2])
    fwhm_error = np.zeros(slices.shape[2])
    rmse = np.zeros(slices.shape[2])
    me = np.zeros(slices.shape[2])
    # parameter = np.zeros(slices.shape[2])

    #
    z_indices = np.arange(slices.shape[2])
    if not np.all(z_include):
        z_indices = z_indices[z_include]

    # Bounds for the fits
    bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, -1], [np.inf, np.inf, np.inf, np.inf, np.inf, 1])
    bounds_n = ([-np.inf, -np.inf, -np.inf, 0, 0, -1, 2], [np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf])

    # for ii in tqdm(range(1215, slices.shape[2])):
    # for ii in tqdm(range(417, 417 + 1)):
    for ii in tqdm(z_indices):
        try:
            if fit_function == 'lorentzian':
                p_opt, c_cov = curve_fit(lorentzian_2d, (x_mesh, y_mesh), slices[:, :, ii].ravel(),
                                         p0=(0.001, 0., 0., 1., 1., 0.), bounds=bounds)
                error = slices[:, :, ii].ravel() - lorentzian_2d((x_mesh, y_mesh), *p_opt)
                factor = np.sqrt(2 ** (2 / 3) - 1)
                # visualize_2d_fit(x, y, slices[:, :, ii], lambda xy: lorentzian_2d(xy, *p_opt), scale='log')

            elif fit_function == 'generalized_lorentzian':
                p_opt, c_cov = curve_fit(generalized_lorentzian_2d, (x_mesh, y_mesh), slices[:, :, ii].ravel(),
                                         p0=[0.001, 0., 0., 1., 1., 0., 3.0], bounds=bounds_n, maxfev=8000)
                # ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, maxfev=0
                error = slices[:, :, ii].ravel() - generalized_lorentzian_2d((x_mesh, y_mesh), *p_opt)
                factor = 1
                # visualize_2d_fit(x, y, slices[:, :, ii], lambda xy: generalized_lorentzian_2d(xy, *p_opt), scale='lin')

            elif fit_function == 'gaussian':
                p_opt, c_cov = curve_fit(gaussian_2d, (x_mesh, y_mesh), slices[:, :, ii].ravel(),
                                         p0=(0.001, 0., 0., 1., 1., 0.), bounds=bounds)
                error = slices[:, :, ii].ravel() - gaussian_2d((x_mesh, y_mesh), *p_opt)
                factor = np.sqrt(2 * np.log(2))
                # visualize_2d_fit(x, y, slices[:, :, ii], lambda xy: gaussian_2d(xy, *p_opt), scale='log')

            else:
                sys.exit('Unknown fit function.')

            # Evaluate the fit
            rmse[ii] = np.sqrt(np.mean(error ** 2))
            me[ii] = np.mean(error)

            # Estimate FWHM
            p_err = np.sqrt(np.diag(c_cov))
            fwhm[ii] = fwhm_2d(p_opt[3], p_opt[4], p_opt[5], factor=factor)
            fwhm_error[ii] = np.sqrt((fwhm_2d(p_opt[3], p_opt[4], p_opt[5], factor=factor, kind='d_gamma_x') * p_err[3]) ** 2 +
                                     (fwhm_2d(p_opt[3], p_opt[4], p_opt[5], factor=factor, kind='d_gamma_y') * p_err[4]) ** 2 +
                                     (fwhm_2d(p_opt[3], p_opt[4], p_opt[5], factor=factor, kind='d_rho') * p_err[5]) ** 2)

            # parameter[ii] = p_opt[0]

        except RuntimeError as err:
            print(err)
            fig, ax = plt.subplots()
            ax.imshow(slices[:, :, ii], extent=get_extent(x, y))
            plt.show()
            pass

    # return fwhm, fwhm_error
    return fwhm, fwhm_error, me


def location_scale_correlation_mapping(xy, x_0, y_0, sigma_x, sigma_y, rho):
    x, y = xy
    x_c = (x - x_0) / sigma_x
    # y_c = (y - y_0) / sigma_y
    y_c = (sigma_x * (y - y_0) - rho * sigma_y * (x - x_0)) / (sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
    norm = sigma_x * sigma_y * np.sqrt(1 - rho ** 2)
    return x_c, y_c, norm


def gaussian_2d(xy, scale, x_0, y_0, sigma_x, sigma_y, rho):
    x_c, y_c, norm = location_scale_correlation_mapping(xy, x_0, y_0, sigma_x, sigma_y, rho)
    z = scale / (2 * np.pi) * np.exp(- (x_c ** 2 + y_c ** 2) / 2) / norm
    return z.ravel()


def lorentzian_2d(xy, scale, x_0, y_0, sigma_x, sigma_y, rho):
    x_c, y_c, norm = location_scale_correlation_mapping(xy, x_0, y_0, sigma_x, sigma_y, rho)
    z = scale / (2 * np.pi) * 1 / (x_c ** 2 + y_c ** 2 + 1) ** (3 / 2) / norm
    return z.ravel()


def generalized_lorentzian_2d(xy, scale, x_0, y_0, sigma_x, sigma_y, rho, n):
    x_c, y_c, norm = location_scale_correlation_mapping(xy, x_0, y_0, sigma_x, sigma_y, rho)
    z = scale / np.pi * np.sin(2 * np.pi / n) / (2 * np.pi / n) * 1 / (np.sqrt(x_c ** 2 + y_c ** 2) ** n + 1) / norm
    return z.ravel()


def check_distribution_normalization(show_distribution=False):
    xx, yy = np.linspace(-20, 20, 1001), np.linspace(-20, 20, 1002)
    # xx, yy = np.linspace(-100, 100, 1001), np.linspace(-100, 100, 1002)
    xx_mesh, yy_mesh = np.meshgrid(xx, yy, indexing='ij')

    # distribution = gaussian_2d((xx_mesh, yy_mesh), 1., 1., -1., 0.5, 2, 0.25).reshape((xx.size, yy.size))
    # distribution = lorentzian_2d((xx_mesh, yy_mesh), 1., 1., -1., 0.5, 1, 0.25).reshape((xx.size, yy.size))
    distribution = generalized_lorentzian_2d((xx_mesh, yy_mesh), 1., 1., -1., 0.5, 1, 0.25, 4.0).reshape((xx.size, yy.size))

    print('∫ f(x,y) dx dy = %1.3f' % np.trapz(np.trapz(distribution, x=xx, axis=0), x=yy))

    if show_distribution:
        fig, ax = plt.subplots()
        ax.imshow(distribution, extent=get_extent(xx, yy))
        plt.show()
    return 0


def fwhm_2d(gamma_x, gamma_y, rho, factor=1, kind=None):
    # fwhm = 2 * np.sqrt(gamma_x * gamma_y * np.sqrt(1 - rho ** 2))
    fwhm = np.sqrt(gamma_x ** 2 + gamma_y ** 2 + 2 * gamma_x * gamma_y * np.sqrt(1 - rho ** 2))

    if kind is None:
        return factor * fwhm
    elif kind == 'd_gamma_x':
        # return factor * 2 * gamma_y * np.sqrt(1 - rho ** 2) / fwhm
        return factor * (gamma_x + gamma_y * np.sqrt(1 - rho ** 2)) / fwhm
    elif kind == 'd_gamma_y':
        # return factor * 2 * gamma_x * np.sqrt(1 - rho ** 2) / fwhm
        return factor * (gamma_y + gamma_x * np.sqrt(1 - rho ** 2)) / fwhm
    elif kind == 'd_rho':
        # return - factor * 2 * gamma_x * gamma_y * rho / (fwhm * np.sqrt(1 - rho ** 2))
        return - factor * gamma_x * gamma_y * rho / (fwhm * np.sqrt(1 - rho ** 2))
    else:
        sys.exit('Unknown kind')


def visualize_2d_fit(x, y, data, model_function, scale='lin'):
    idx_0 = np.argmin(np.abs(x))

    xx = np.linspace(x[0], x[-1], 101)
    yy = np.linspace(x[0], x[-1], 101)
    model = model_function(np.meshgrid(xx, yy)).reshape((xx.size, yy.size))

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 6.2))
    gs = fig.add_gridspec(2, 4, width_ratios=(1, 3, 0.4, 0.2), height_ratios=(1, 3),
                          left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.1, hspace=0.05)
    ax = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[0, 1], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 0], sharey=ax)
    ax_ = fig.add_subplot(gs[1, 3])

    if scale == 'log':
        levels = np.geomspace(np.min(data), np.max(data), 5 + 1)
        levels = (levels[1:] + levels[:-1]) / 2
        im = ax.imshow(data.T, origin='lower', extent=get_extent(x, y), norm=LogNorm())
        cn = ax.contour(xx, yy, model, levels, cmap='Reds', norm=LogNorm())
    else:
        levels = np.linspace(np.min(data), np.max(data), 5 + 1)
        levels = (levels[1:] + levels[:-1]) / 2
        im = ax.imshow(data.T, origin='lower', extent=get_extent(x, y))
        cn = ax.contour(xx, yy, model, levels, cmap='Reds')

    ax.set_xlabel(r'$x$ [mm]')
    ax.set_ylabel(r'$y$ [mm]')
    ax.yaxis.set_label_position('right')
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.tick_params(axis='y', which='both', left=False, labelleft=False, right=True, labelright=True)

    c_bar = fig.colorbar(im, cax=ax_, orientation='vertical')
    c_bar.add_lines(cn)
    c_bar.set_label('"Activity"')

    line_width = 2
    ax_x.plot(xx, np.trapz(model, x=yy, axis=1), linewidth=line_width, color='tab:orange')
    ax_x.plot(x, np.sum(data, axis=1), linewidth=line_width, color='tab:blue', linestyle='--')
    ax_x.plot(xx, model[:, 50], linewidth=line_width, color='tab:red')
    ax_x.plot(x, data[:, idx_0], linewidth=line_width, color='tab:green', linestyle='--')
    ax_x.tick_params(axis='x', which='both', labelbottom=False)

    ax_y.plot(np.trapz(model, x=xx, axis=0), yy, linewidth=line_width, color='tab:orange')
    ax_y.plot(np.sum(data, axis=0), y, linewidth=line_width, color='tab:blue', linestyle='--')
    ax_y.plot(model[50, :], yy, linewidth=line_width, color='tab:red')
    ax_y.plot(data[idx_0, :], y, linewidth=line_width, color='tab:green', linestyle='--')
    ax_y.tick_params(axis='y', which='both', left=False, labelleft=False, right=True)
    ax_y.invert_xaxis()

    if scale == 'log':
        ax_x.set_yscale('log')
        ax_y.set_xscale('log')

    else:
        ax_x.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_y.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        c_bar.formatter.set_scientific(True)
        c_bar.formatter.set_powerlimits((0, 0))

    plt.show()

    return 0


def compare_fit_error(z, gauss_list, lorentz_list, label_list):

    plt.rcParams.update({'font.size': 16})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots()
    for ii in range(len(label_list)):
        ax.plot(z, gauss_list[ii], color=colors[ii], label=label_list[ii] + ' (Gauss)')
        ax.plot(z, lorentz_list[ii], linestyle='--', color=colors[ii], label=label_list[ii] + ' (Lorentz)')

    ax.set_xlabel(r'$z$ [mm]')
    ax.set_ylabel('Mean error')
    ax.legend(ncol=2)

    plt.show()
    return 0


if __name__ == "__main__":
    main()
