"""
Fit Modulation Transfer Functions (MTFs) to contrast data of Point Spread Functions (PSFs) to images


Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.optimize import curve_fit

# Auxiliary functions
from psf_mtf_library import get_mtf


def main():
    return 0


def fit_mtf(wave_numbers, contrast_values, contrast_errors, model='gaussian', include_amplitude=False):

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


def fit_psf_2d():

    return 0


if __name__ == "__main__":
    main()
