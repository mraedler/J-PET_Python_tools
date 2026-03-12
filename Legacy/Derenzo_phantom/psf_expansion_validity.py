"""
Set of point spread functions in 2D

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from matplotlib import pyplot as plt


def main():
    lim = 2.5
    alpha_edges = np.linspace(-lim, lim, 1000)
    beta_edges = np.linspace(-lim, lim, 1000)
    extent = [alpha_edges[0], alpha_edges[-1], beta_edges[0], beta_edges[-1]]
    alpha, beta = np.meshgrid((alpha_edges[1:] + alpha_edges[:-1]) / 2, (beta_edges[1:] + beta_edges[:-1]) / 2, indexing='ij')

    # 2D case
    # rad = alpha ** 2 - 2 * beta + beta ** 2
    # x_0 = - np.sqrt(1 - (alpha - np.sqrt(rad)) / beta)
    # x_1 = - np.sqrt(1 - (alpha + np.sqrt(rad)) / beta)
    # x_2 = np.sqrt(1 - (alpha - np.sqrt(rad)) / beta)
    # x_3 = np.sqrt(1 - (alpha + np.sqrt(rad)) / beta)



    fig, ax = plt.subplots()

    # ax.plot(alpha_edges, alpha_valid(alpha_edges))
    ax.imshow(alpha_beta_valid(alpha, beta).T, origin='lower', extent=extent)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    plt.show()

    return 0


def alpha_valid(alpha):
    # Cast to ndarray, if not already
    if not isinstance(alpha, np.ndarray):
        alpha = np.array(alpha, ndmin=1)

    valid = np.zeros(alpha.shape, dtype=bool)
    valid[(alpha >= 0) & (alpha <= 1)] = True

    return valid


def alpha_beta_valid(alpha, beta):
    # Cast to ndarray, if not already
    if not (isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray)):
        alpha, beta = np.array(alpha, ndmin=1), np.array(beta, ndmin=1)

    singularity_entry = (alpha ** 2 - 2 * beta + beta ** 2)
    singularity_entry[alpha != beta] /= (alpha - beta)[alpha != beta]

    minus_inf = np.stack((beta, -beta, beta - alpha, singularity_entry, 1 - alpha), axis=alpha.ndim)
    plus_inf = np.stack((beta, beta, beta - alpha, -singularity_entry, 1 - alpha), axis=alpha.ndim)

    n_minus_inf = np.sum(np.abs(np.diff(half_sign(minus_inf), axis=-1)), axis=-1)
    n_plus_inf = np.sum(np.abs(np.diff(half_sign(plus_inf), axis=-1)), axis=-1)

    n_zeros = n_minus_inf - n_plus_inf

    valid = np.zeros(n_zeros.shape, dtype=bool)
    valid[n_zeros == 0.] = True

    # return n_zeros
    return valid


def half_sign(x, zero_is_positive=False):
    y = np.zeros(x.shape)
    if zero_is_positive:
        y[x >= 0] = 0.5
        y[x < 0] = -0.5
    else:
        y[x > 0] = 0.5
        y[x <= 0] = -0.5
    return y


def alpha_beta_range():

    alpha_edges = np.linspace(-3, 3, 201)
    beta_edges = np.linspace(-1, 3, 203)

    alpha_centers = (alpha_edges[1:] + alpha_edges[:-1]) / 2
    beta_centers = (beta_edges[1:] + beta_edges[:-1]) / 2

    print(beta_centers)

    alpha_mesh, beta_mesh = np.meshgrid(alpha_centers.astype(np.complex128), beta_centers.astype(np.complex128), indexing='ij')

    x0 = x_0_zero_crossing(alpha_mesh, beta_mesh)
    x1 = x_1_zero_crossing(alpha_mesh, beta_mesh)
    x2 = x_0_zero_crossing2(alpha_mesh, beta_mesh)


    x0_not_imaginary = (np.imag(x0) != 0)
    x1_not_imaginary = (np.imag(x1) != 0)
    x2_not_imaginary = (np.imag(x2) != 0)

    aa = (x0_not_imaginary) & (x1_not_imaginary) & (x2_not_imaginary)


    fig, ax = plt.subplots()
    ax.imshow(aa.T, origin='lower', extent=(alpha_edges[0], alpha_edges[-1], beta_edges[0], beta_edges[-1]))
    plt.show()

    return 0


def x_0_zero_crossing(alpha_mesh, beta_mesh):
    x_0 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_0[~beta_0] = np.sqrt(-alpha_mesh[~beta_0] / beta_mesh[~beta_0] + np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0]) / beta_mesh[~beta_0])
    x_0[beta_0] = 1j / np.sqrt(alpha_mesh[beta_0])

    return x_0


def x_1_zero_crossing(alpha_mesh, beta_mesh):
    x_1 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_1[~beta_0] = np.sqrt(-alpha_mesh[~beta_0] / beta_mesh[~beta_0] - np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0]) / beta_mesh[~beta_0])
    x_1[beta_0] = 1j * np.inf

    return x_1


def x_0_zero_crossing2(alpha_mesh, beta_mesh):
    x_0 = np.zeros(alpha_mesh.shape, dtype=np.complex128)
    beta_0 = np.abs(beta_mesh) < 1e-12

    x_0[~beta_0] = np.sqrt(1 - alpha_mesh[~beta_0] / beta_mesh[~beta_0] + np.sqrt(alpha_mesh[~beta_0] ** 2 - 2 * beta_mesh[~beta_0] + beta_mesh[~beta_0] ** 2) / beta_mesh[~beta_0])
    # x_0[beta_0] = 1j / np.sqrt(alpha_mesh[beta_0])

    return x_0


if __name__ == "__main__":
    main()
    alpha_beta_range()
