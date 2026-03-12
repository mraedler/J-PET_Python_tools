"""
Simplify the analytical expression of the 2D PSF originating from the plateau-polynomial MTF

Author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
from scipy.special import j0, j1, jv, struve
from scipy.integrate import trapezoid
from matplotlib import pyplot as plt


def main():
    k_50 = 1.12
    alpha = 0.48

    a = (1 - alpha) * k_50
    b = (1 + alpha) * k_50

    x = np.linspace(-200, 200, 1000)
    y = np.linspace(-200, 200, 1000)

    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    rho = np.sqrt(x_mesh ** 2 + y_mesh ** 2)



    psf_2d_plateau = 1 / (2 * np.pi) * bessel_integral_1(rho, a)
    psf_1d_plateau = inverse_fourier_0(x, a)

    const_0 = 1 / 4 * (2 - 1 / alpha ** 3 + 3 / alpha)
    pdf_2d_0 = const_0 / (2 * np.pi) * (bessel_integral_1(rho, b) - bessel_integral_1(rho, a))
    psf_1d_0 = const_0 * (inverse_fourier_0(x, b) - inverse_fourier_0(x, a))

    const_1 = - 3 / 4 * (alpha ** 2 - 1) / (k_50 * alpha ** 3)
    pdf_2d_1 = const_1 / (2 * np.pi) * (bessel_integral_2(rho, b) - bessel_integral_2(rho, a))
    psf_1d_1 = const_1 * (inverse_fourier_1(x, b) - inverse_fourier_1(x, a))

    const_2 = - 3 / 4 * 1 / (k_50 ** 2 * alpha ** 3)
    pdf_2d_2 = const_2 / (2 * np.pi) * (bessel_integral_3(rho, b) - bessel_integral_3(rho, a))
    psf_1d_2 = const_2 * (inverse_fourier_2(x, b) - inverse_fourier_2(x, a))

    const_3 = 1 / (4 * k_50 ** 3 * alpha ** 3)
    pdf_2d_3 = const_3 / (2 * np.pi) * (bessel_integral_4(rho, b) - bessel_integral_4(rho, a))
    psf_1d_3 = const_3 * (inverse_fourier_3(x, b) - inverse_fourier_3(x, a))



    # kk = np.linspace(a, b, 100)
    # fig, ax = plt.subplots()
    # ax.plot(kk, const_0 + const_1 * kk + const_2 * kk ** 2 + const_3 * kk ** 3)
    # plt.show()

    psf_2d = psf_2d_plateau + pdf_2d_0 + pdf_2d_1 + pdf_2d_2 + pdf_2d_3
    psf_1d = psf_1d_plateau + psf_1d_0 + psf_1d_1 + psf_1d_2 + psf_1d_3
    xi = k_50 * x
    psf_1d_alt = k_50 / np.pi * np.sin(xi) / xi * (np.sin(alpha * xi) - alpha * xi * np.cos(alpha * xi)) / (alpha **3 * xi ** 3 / 3)


    # psf_2d_min_1 = 3 / 8 * (1 - alpha) ** 2 / alpha ** 3 * ((1 - alpha) * jv(0, (1 - alpha) * k_50 * rho) - (1 + 3 * alpha) * jv(2, (1 - alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_2 = - 3 / 8 * (1 + alpha) ** 2 / alpha ** 3 * ((1 + alpha) * jv(0, (1 + alpha) * k_50 * rho) + (-1 + 3 * alpha) * jv(2, (1 + alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_3 = 3 / 4 * (3 + k_50 ** 2 * (alpha ** 2 - 1) * rho ** 2) / (k_50 ** 3 * alpha ** 3 * rho ** 2) * (bessel_integral_2_alt(rho, a) - bessel_integral_2_alt(rho, b))
    #
    # bbb = (psf_2d_min_1 + psf_2d_min_2 + psf_2d_min_3) / (2 * np.pi)
    # # print(trapezoid(pdf_1d_integrated, x=x))
    #
    # psf_2d_min_min_1 = 3 / 8 * (1 - alpha) ** 2 / alpha ** 3 * ( - (1 + 3 * alpha) * jv(2, (1 - alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_2 = - 3 / 8 * (1 + alpha) ** 2 / alpha ** 3 * ( + (-1 + 3 * alpha) * jv(2, (1 + alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_3 = 3 / 4 * 3 / (k_50 ** 3 * alpha ** 3 * rho ** 2) * (bessel_integral_2_alt_alt(rho, a) - bessel_integral_2_alt_alt(rho, b))
    # psf_2d_min_min_4 = 3 / 4 * (alpha ** 2 - 1) / (k_50 * alpha ** 3) * (bessel_integral_2_alt(rho, a) - bessel_integral_2_alt(rho, b))
    #
    # ccc = (psf_2d_min_min_1 + psf_2d_min_min_2 + psf_2d_min_min_3 + psf_2d_min_min_4) / (2 * np.pi)
    #
    # psf_2d_min_min_min_1 = - 3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * ((1 - alpha) * k_50 * jv(2, (1 - alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_min_2 = 3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * ((1 + alpha) * k_50 * jv(2, (1 + alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_min_3 = 3 / 4 * 3 / (k_50 ** 3 * alpha ** 3 * rho ** 2) * (bessel_integral_2_alt_alt_alt(rho, a) - bessel_integral_2_alt_alt_alt(rho, b))
    # psf_2d_min_min_min_4 = -3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * (bessel_integral_2_alt(rho, a) - bessel_integral_2_alt(rho, b))
    #
    # ddd = (psf_2d_min_min_min_1 + psf_2d_min_min_min_2 + psf_2d_min_min_min_3 + psf_2d_min_min_min_4) / (2 * np.pi)

    # psf_2d_min_min_min_1 = - 3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * ((1 - alpha) * k_50 * jv(2, (1 - alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_min_2 = 3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * ((1 + alpha) * k_50 * jv(2, (1 + alpha) * k_50 * rho)) / rho ** 2
    # psf_2d_min_min_min_min_3 = 3 / 4 * 3 / (k_50 ** 3 * alpha ** 3 * rho ** 2) * (bessel_integral_2_alt_alt_alt(rho, a) - bessel_integral_2_alt_alt_alt(rho, b))
    # psf_2d_min_min_min_min_4 = -3 / 4 * (1 - alpha ** 2) / (k_50 * alpha ** 3) * (bessel_integral_2_alt_alt_alt_alt(rho, a) - bessel_integral_2_alt_alt_alt_alt(rho, b))
    #
    # eee = (psf_2d_min_min_min_min_3 + psf_2d_min_min_min_min_4) / (2 * np.pi)

    psf_2d_one = (1 - alpha ** 2) * (bessel_integral_f(b * rho) - bessel_integral_f(a * rho))
    psf_2d_two = 3 / (k_50 ** 2 * rho ** 2) * (bessel_integral_g(b * rho) - bessel_integral_g(a * rho))

    fff = 3 / 4 * 1 / (alpha ** 3 * rho ** 2) * 1 / (k_50 * rho) * (psf_2d_one - psf_2d_two) / (2 * np.pi)

    fig, ax = plt.subplots()
    # ax.imshow(psf_2d.T, origin='lower')
    # ax.plot(x, psf_1d)
    ax.plot(x, psf_1d_alt)
    ax.plot(x, trapezoid(psf_2d, x=y, axis=1), linestyle='--')
    # ax.plot(x, trapezoid(bbb, x=y, axis=1), linestyle=':')
    # ax.plot(x, trapezoid(ccc, x=y, axis=1), linestyle=':')
    # ax.plot(x, trapezoid(ddd, x=y, axis=1), linestyle=':')
    # ax.plot(x, trapezoid(eee, x=y, axis=1), linestyle=':')
    ax.plot(x, trapezoid(fff, x=y, axis=1), linestyle=':')
    # ax.plot(x, trapezoid(psf_2d_plateau, x=y, axis=1), linestyle=':')
    plt.show()

    return 0


"""2D"""


def bessel_integral_1(rho, b):
    return b * j1(b * rho) / rho


def bessel_integral_2(rho, b):
    return b * bessel_integral_1(rho, b) + b * np.pi / 2 * (j0(b * rho) * struve(1, b * rho) - j1(b * rho) * struve(0, b * rho)) / rho ** 2


def bessel_integral_2_alt(rho, b):
    return b * np.pi / 2 * (j0(b * rho) * struve(1, b * rho) - j1(b * rho) * struve(0, b * rho)) / rho ** 2


def bessel_integral_2_alt_alt(rho, b):
    return b ** 2 * np.pi / 4 * (j0(b * rho) * struve(2, b * rho) - jv(2, b * rho) * struve(0, b * rho)) / rho


def bessel_integral_2_alt_alt_alt(rho, b):
    return b * np.pi / 2 * (jv(1, b * rho) * struve(2, b * rho) - jv(2, b * rho) * struve(1, b * rho)) / rho ** 2


def bessel_integral_2_alt_alt_alt_alt(rho, b):
    return b * np.pi / 2 * (jv(0, b * rho) * struve(1, b * rho) - jv(1, b * rho) * struve(0, b * rho)) / rho ** 2 + b / rho ** 2 * jv(2, b * rho)


def bessel_integral_3(rho, b):
    return b ** 2 * (2 * jv(2, b * rho) - b * rho * jv(3, b * rho)) / rho ** 2


def bessel_integral_4(rho, b):
    return b ** 3 * (bessel_integral_1(rho, b) + 3 * j0(b * rho) / rho ** 2) - 9 / rho ** 2 * bessel_integral_2(rho, b)


"""2D Final"""


def bessel_integral_f(x):
    return bessel_integral_h(x) + x * jv(2, x)


def bessel_integral_g(x):
    # return np.pi * x * (jv(1, x) * struve(2, x) - jv(2, x) * struve(1, x)) / 2
    return bessel_integral_h(x) + x ** 2 * jv(1, x) / 3


def bessel_integral_h(x):
    return np.pi * x * (jv(0, x) * struve(1, x) - jv(1, x) * struve(0, x)) / 2


"""1D"""


def inverse_fourier_0(x, b):
    return np.sin(b * x) / (np.pi * x)


def inverse_fourier_1(x, b):
    bx = b * x
    return (np.cos(bx) + bx * np.sin(bx)) / (np.pi * x ** 2)


def inverse_fourier_2(x, b):
    bx = b * x
    return (2 * bx * np.cos(bx) + (bx ** 2 - 2) * np.sin(bx)) / (np.pi * x ** 3)


def inverse_fourier_3(x, b):
    bx = b * x
    return (3 * (bx ** 2 - 2) * np.cos(bx) + bx * (bx ** 2 - 6) * np.sin(bx)) / (np.pi * x ** 4)


if __name__ == "__main__":
    main()
