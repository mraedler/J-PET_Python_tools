"""
Fit Modulation Transfer Functions (MTFs) to contrast data of Point Spread Functions (PSFs) to images


Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d, fftconvolve
from scipy.optimize import curve_fit, minimize
from scipy.integrate import cumulative_trapezoid
from tqdm import trange

# Auxiliary functions
from Derenzo_phantom.psf_mtf_library import get_mtf, psf_hermite_gaussian_2d, psf_plateau_polynomial_2d, fwhm_hermite_gaussian_1d, fwhm_plateau_polynomial_1d


def main():
    return 0


def fit_mtf(wave_numbers, contrast_values, contrast_errors, model='hermite-gaussian', include_amplitude=False):

    p0, lower_bounds, upper_bounds, mtf_model = get_mtf(model)

    if include_amplitude:
        def fit_function(*args):
            *func_args, amplitude = args
            return mtf_model(*func_args) * amplitude

        p0.append(1.)
        lower_bounds.append(-1.)
        upper_bounds.append(1.)

    else:
        def fit_function(*args):
            return mtf_model(*args)

    # # noinspection PyTupleAssignmentBalance
    # p_opt, p_cov = curve_fit(fit_function, wave_numbers, contrast_values,
    #                          p0=p0,
    #                          bounds=(lower_bounds,
    #                                  upper_bounds),
    #                          sigma=contrast_errors,
    #                          absolute_sigma=True,
    #                          maxfev=10000)

    # noinspection PyTupleAssignmentBalance
    p_opt, p_cov = curve_fit(fit_function, wave_numbers, contrast_values,
                             p0=p0,
                             bounds=(lower_bounds,
                                     upper_bounds),
                             maxfev=10000)

    p_err = np.sqrt(np.diag(p_cov))

    return lambda k: fit_function(k, *p_opt), p_opt, p_err


def fit_psf_2d(x_grid_psf, y_grid_psf, img_recon, img_derenzo, model='hermite-gaussian', show_convergence=False):
    """
    :param x_grid_psf: The mid-points between the grid values have to go through the center x=0 for a symmetric kernel
    :param y_grid_psf: The mid-points between the grid values have to go through the center y=0 for a symmetric kernel
    """
    # evaluation_vs_integration(x_grid_psf, y_grid_psf, model)

    p_0 = np.array([1, 0.5])
    p_s = np.array([1, 0.1])

    # Keep track of the cost function values
    parameters_values_dict = {}
    evaluation_counter = 0

    def objective_function(params):
        # todo: punish alpha outside
        nonlocal evaluation_counter
        # Root mean square error
        value = np.sqrt(np.mean((img_recon - params[0] * blur_ground_truth(x_grid_psf, y_grid_psf, img_derenzo, model, params[1:], n=150)) ** 2))

        # Keep track of the pairs of parameters and their objective function values
        parameters_values_dict[tuple(params)] = value

        evaluation_counter += 1
        print('Evaluation #%d: %1.3e' % (evaluation_counter, value), end="\r", flush=True)
        return value

    # Initial guess
    amplitude_0 = np.sum(img_recon) / np.sum(img_derenzo)
    x_0 = np.insert(p_0, 0, amplitude_0)
    x_s = np.insert(p_s, 0, amplitude_0)
    initial_simplex = np.tile(x_0, (x_0.size + 1, 1))
    initial_simplex[1:, :] += np.diag(x_s)

    # Run the optimization
    p_opt = minimize(objective_function, method='Nelder-Mead', x0=x_0, options={'initial_simplex': initial_simplex, 'return_all': True, 'maxfev': 10000})
    print()
    # todo: Add parameter progression

    if show_convergence:
        show_convergence_optimization(p_opt.allvecs, parameters_values_dict)

    return p_opt


def evaluation_vs_integration(x_grid_psf, y_grid_psf, model):
    # Investigate the differences between
    # - evaluating the PDF at regular grid points and weighing them with the pixel size
    # - integrating each pixel individually

    if model == 'hermite-gaussian':
        def point_spread_function(xx, yy): return psf_hermite_gaussian_2d(xx, yy, 0.5, 0.5)

    elif model == 'plateau-polynomial':
        def point_spread_function(xx, yy): return psf_plateau_polynomial_2d(xx, yy, 7, 0.5)
    else:
        sys.exit('Unknown model.')

    # Evaluation approach
    x_psf = (x_grid_psf[:-1] + x_grid_psf[1:]) / 2
    y_psf = (y_grid_psf[:-1] + y_grid_psf[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_psf, y_psf, indexing='ij')
    pmf_eval = point_spread_function(x_mesh, y_mesh) * (x_psf[1] - x_psf[0]) * (y_psf[1] - y_psf[0])
    pmf_eval_marginal = np.sum(pmf_eval, axis=1)

    _, x_marginal, pmf_intg_marginal = numerical_kernel_integral(x_grid_psf, y_grid_psf, pdf_function=point_spread_function)

    print('Evaluation approach:  %1.10f' % np.sum(pmf_eval_marginal))
    print('Integration approach: %1.10f' % np.sum(pmf_intg_marginal))

    fig, ax = plt.subplots()
    ax.plot(x_psf, pmf_eval_marginal, label='Evaluated')
    ax.plot(x_marginal, pmf_intg_marginal, label='Integrated')
    ax.legend()
    plt.show()

    return 0


def blur_ground_truth(x_grid_psf, y_grid_psf, img_ground_truth, model, params, n=0):
    if model == 'hermite-gaussian':
        point_spread_function, _, _ = numerical_kernel_integral(x_grid_psf[n:-n or None], y_grid_psf[n:-n or None], pdf_function=lambda xx, yy: psf_hermite_gaussian_2d(xx, yy, *params))
    elif model == 'plateau-polynomial':
        point_spread_function, _, _ = numerical_kernel_integral(x_grid_psf[n:-n or None], y_grid_psf[n:-n or None], pdf_function=lambda xx, yy: psf_plateau_polynomial_2d(xx, yy, *params))
    else:
        sys.exit('Unrecognized model: %s' % model)

    # point_spread_function[point_spread_function < 0] = 0
    # point_spread_function /= point_spread_function.sum()
    print(point_spread_function.sum())

    # return convolve2d(img_ground_truth, point_spread_function, mode='same')
    return fftconvolve(img_ground_truth, point_spread_function, mode='same')


def numerical_kernel_integral(x_grid, y_grid, pdf_function, n_upscale=3):
    # Upscale the samples
    x_grid_up, idx_x_up = upscale_samples(x_grid, n_upscale)
    y_grid_up, idx_y_up = upscale_samples(y_grid, n_upscale)

    # Evaluate the PDF
    x_mesh, y_mesh = np.meshgrid(x_grid_up, y_grid_up, indexing='ij')
    pdf = pdf_function(x_mesh, y_mesh)
    # print(np.trapz(np.trapz(pdf, x=x_grid_up, axis=0), x=y_grid_up))

    # Integrate numerically
    itg_x_up = cumulative_trapezoid(pdf, x=x_grid_up, axis=0, initial=0)
    itg_x = itg_x_up[idx_x_up[1:], :] - itg_x_up[idx_x_up[:-1], :]

    itg_x_y_up = cumulative_trapezoid(itg_x, x=y_grid_up, axis=1, initial=0)
    itg_x_y = itg_x_y_up[:, idx_y_up[1:]] - itg_x_y_up[:, idx_y_up[:-1]]
    # print(np.sum(itg_x_y))

    x_marginal = (x_grid[:-1] + x_grid[1:]) / 2
    pmf_marginal = itg_x_y_up[:, -1]

    return itg_x_y, x_marginal, pmf_marginal


def upscale_samples(x, n):
    x_new = x.copy()
    for ii in range(n):
        x_mid = (x_new[1:] + x_new[:-1]) / 2
        idx_mid = np.arange(1, x_new.size)
        x_new = np.insert(x_new, idx_mid, x_mid)

    idx_initial = np.arange(0, x_new.size, 2 ** n)

    # Consistency check
    # print(x_new[idx_initial] - x)

    return x_new, idx_initial


def show_convergence_optimization(all_vecs, parameters_values_dict):
    obj_fun_itr = np.array([parameters_values_dict[tuple(v)] for v in all_vecs])

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


if __name__ == "__main__":
    main()
