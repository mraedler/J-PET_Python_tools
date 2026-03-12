"""
Optimize the PSF to best describe the blurring of the reconstruction

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Auxiliary functions
from point_spread_functions_2d import get_2d_point_spread_function


def main():
    return 0


def run_optimization(x_grid, y_grid, img_recon, img_derenzo, kind, visualize=False, verbose=False):
    #
    psf, fwhm, p_0, p_s, fourier_profile = get_2d_point_spread_function(kind, return_fourier_profile=True)

    # For the kernel
    x = (x_grid[1:] + x_grid[:-1]) / 2
    y = (y_grid[1:] + y_grid[:-1]) / 2

    def blur_derenzo(params):
        point_spread_function, _, _ = psf(x, y, params)
        # return convolve2d(img_derenzo, point_spread_function, mode='same')
        return fftconvolve(img_derenzo, point_spread_function, mode='same')

    def objective_function(params):
        return np.sum((img_recon - params[0] * blur_derenzo(params[1:])) ** 2)

    # Initial guess
    alpha_0 = np.sum(img_recon) / np.sum(img_derenzo)
    x_0 = np.insert(p_0, 0, alpha_0)
    x_s = np.insert(p_s, 0, alpha_0)
    initial_simplex = np.tile(x_0, (x_0.size + 1, 1))
    initial_simplex[1:, :] += np.diag(x_s)

    # Optimization
    p_opt = minimize(objective_function, method='Nelder-Mead', x0=x_0, options={'initial_simplex': initial_simplex, 'return_all': True, 'maxfev': 10000})  #
    # p_opt = minimize(objective_function, method='BFGS', x0=x_0)
    # print(np.sqrt(np.diag(p_opt.hess_inv)))
    if verbose:
        print(p_opt.message)
        print('Objective function: %1.2f' % (p_opt.fun / 1e5))
        print('FWHM: %1.3f' % fwhm(p_opt.x[1:]))

    img_derenzo_convolved = p_opt.x[0] * blur_derenzo(p_opt.x[1:])
    # img_derenzo_convolved = p_opt.x[0] * img_derenzo
    psf_2d, x_1d, psf_1d = psf(x, y, p_opt.x[1:])

    # fig, ax = plt.subplots()
    # ax.plot(x_grid[1:-1], psf_2d[79, :])
    # ax.plot(x_1d, psf_1d)
    # plt.show()

    if visualize:
        show_convergence_optimization(p_opt.allvecs, objective_function)

        c_lim = [np.min(img_recon), np.max(img_recon)]
        extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
        plt.rcParams.update({'font.size': 12})
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 6))
        ax0.imshow(img_recon.T, origin='lower', extent=extent, clim=c_lim)
        ax0.set_xlabel(r'$x$ [mm]')
        ax0.set_ylabel(r'$y$ [mm]')
        ax0.set_title('Reconstruction')
        ax1.imshow(img_derenzo_convolved.T, origin='lower', extent=extent, clim=c_lim)
        ax1.set_xlabel(r'$x$ [mm]')
        # ax1.set_ylabel(r'$y$ [mm]')
        ax1.set_title('Convolved ground truth')
        ax2.imshow(p_opt.x[0] * img_derenzo.T, origin='lower', extent=extent, clim=c_lim)
        ax2.set_xlabel(r'$x$ [mm]')
        # ax2.set_ylabel(r'$y$ [mm]')
        ax2.set_title('Ground truth')
        plt.show()

    return p_opt, fwhm, objective_function, blur_derenzo, fourier_profile


def show_convergence_optimization(all_vecs, objective_function):
    # Evaluate the
    obj_fun_itr = np.zeros(len(all_vecs))
    for ii in trange(len(all_vecs)):
        obj_fun_itr[ii] = objective_function(all_vecs[ii])

    it = np.arange(1, len(all_vecs) + 1)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(it, obj_fun_itr, color='tab:blue')
    ax.set_xlim(it[0] - 1, it[-1] + 1)
    ax.tick_params(axis='y', which='both', colors='tab:blue')
    # ax.spines['left'].set_color('tab:blue')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel(r'Iteration number $i$')
    ax.set_ylabel(r'Cost function $C(p_i)$', color='tab:blue')

    ax_twin = ax.twinx()
    ax_twin.plot(it, obj_fun_itr - obj_fun_itr[-1], color='tab:orange')
    ax_twin.set_yscale('log')
    ax_twin.tick_params(axis='y', which='both', colors='tab:orange')
    ax_twin.spines['left'].set_color('tab:blue')
    ax_twin.spines['right'].set_color('tab:orange')
    ax_twin.set_ylabel(r'$C(p_i) - C(p_{i_\mathrm{max}})$', color='tab:orange')

    fig.tight_layout()

    plt.show()
    return 0


def compare_derenzo_line_profiles(x_grid, y_grid, img_0, img_1, x_out, y_out, idx):
    # Use them as midpoint and directions
    x_mid = ((x_out[:, 0] + x_out[:, 1]) / 2)[np.newaxis, :]
    y_mid = ((y_out[:, 0] + y_out[:, 1]) / 2)[np.newaxis, :]

    x_vec = (x_out[:, 1] - x_out[:, 0])[np.newaxis, :]
    y_vec = (y_out[:, 1] - y_out[:, 0])[np.newaxis, :]
    norm = np.sqrt(x_vec ** 2 + y_vec ** 2)
    x_vec, y_vec = x_vec / norm, y_vec / norm

    # Interpolation points
    # t = np.linspace(-18, 18, 100)[:, np.newaxis]
    t = np.linspace(-26, 26, 100)[:, np.newaxis]

    x_itp = (x_mid + t * x_vec)[:, idx]
    y_itp = (y_mid + t * y_vec)[:, idx]
    points = np.stack((x_itp.flatten(), y_itp.flatten()), axis=1)

    # Run the interpolation
    x, y = (x_grid[1:] + x_grid[:-1]) / 2, (y_grid[1:] + y_grid[:-1]) / 2
    interpolator_0 = RegularGridInterpolator((x, y), img_0)
    itp_img_0 = interpolator_0(points).reshape(x_itp.shape)

    interpolator_1 = RegularGridInterpolator((x, y), img_1)
    itp_img_1 = interpolator_1(points).reshape(x_itp.shape)

    # Plot
    plt.rcParams.update({'font.size': 16})
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 6))
    fig = plt.figure(figsize=(8, 6.5))
    gs = GridSpec(2, 2, height_ratios=[(1 + np.sqrt(5))/2, 1])
    ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    ax0.imshow(img_0.T, origin='lower', extent=extent)
    ax0.plot(x_itp, y_itp, '--', color='tab:blue')
    ax0.set_xlabel(r'$x$ [mm]')
    ax0.set_ylabel(r'$y$ [mm]')
    # ax0.set_title('Reconstructed image')
    ax0.set_title(r'$6\times30$ mm$^2$')
    # ax0.set_xticks([-80, -40, 0, 40, 80])
    # ax0.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])

    ax1.imshow(img_1.T, origin='lower', extent=extent, vmin=np.min(img_0), vmax=np.max(img_0))
    # ax1.scatter(x_out, y_out, color='tab:red')
    # ax1.scatter(x_mid, y_mid, color='tab:red')
    # ax1.quiver(x_mid, y_mid, x_vec, y_vec, scale=1/18, scale_units='xy', color='tab:red')
    ax1.plot(x_itp, y_itp, '--', color='tab:orange')
    ax1.set_xlabel(r'$x$ [mm]')
    # ax1.set_ylabel(r'$y$ [mm]')
    ax1.set_yticks([])
    # ax1.set_title('Convolved ground truth')
    ax1.set_title(r'$4\times18$ mm$^2$')


    ax2.plot(t, itp_img_0, color='tab:blue', label='Reconstruction')
    ax2.plot(t, itp_img_1, color='tab:orange', linestyle='--', label='Convolved ground truth')
    # f = 1 / (4 * 2)
    # ax2.plot(t, 1.5e2 * (1 - np.cos(2 * np.pi * f * t)) / 2, color='tab:green')
    # ax2.set_ylim(np.min(img_0), np.max(img_0))
    ax2.set_ylim(-0.05e2, 1.05e2)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax2.set_aspect(np.diff(ax2.get_xlim()) / np.diff(ax2.get_ylim()))
    # ax2.set_xlabel('Distance [mm]')
    ax2.set_xlabel('Profile [mm]')
    ax2.set_ylabel('Intensity [arb. units]')
    ax2.set_title('Profile')
    ax2.legend(ncol=2, frameon=False)

    plt.show()
    return 0


if __name__ == "__main__":
    main()
