"""
Convenience functions for printing the significant digits of given values and the corresponding error

Author: Martin Rädler
"""
# Python libraries
import numpy as np


def print_value_and_error(value, error):
    exp_error = np.floor(np.log10(np.abs(error)))
    error_rounded = round_to_exponent(error, exp_error)
    exp_error_rounded = np.floor(np.log10(np.abs(error_rounded)))
    error_rounded = round_to_exponent(error_rounded, exp_error_rounded)
    value_rounded = round_to_exponent(value, exp_error_rounded)

    format_specifier = '%1.5f ± %1.5f  ⇒  %1.' + ('%s' % int(abs(exp_error_rounded))) + 'f(%d)'
    print(format_specifier % (value, error, value_rounded, int(error_rounded / 10 ** exp_error_rounded)))

    return 0


def round_to_exponent(value, exponent):
    return np.round(value / 10 ** exponent) * 10 ** exponent


def error_propagation(function, inputs, inputs_error):

    derivatives = finite_differences(function, inputs)
    gaussian_error = np.sqrt(np.sum(derivatives ** 2 * inputs_error ** 2))

    return gaussian_error


def finite_differences(function, inputs, delta=0.001):

    derivatives = np.zeros(inputs.size)

    for ii in range(inputs.size):
        inputs_forward = inputs + np.eye(1, inputs.size, k=ii).flatten() * delta
        inputs_backward = inputs - np.eye(1, inputs.size, k=ii).flatten() * delta

        derivatives[ii] = (function(*inputs_forward) - function(*inputs_backward)) / (2 * delta)

    return derivatives
