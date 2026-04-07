"""

"""
import sys
import numpy as np
from scipy.optimize import curve_fit
from tqdm import trange
import matplotlib.pyplot as plt


def main():
    ell = 10
    tau_m = 2
    tau_p = 3

    l = np.linspace(-ell, ell, 201)
    t = np.linspace(-max(tau_m, tau_p), max(tau_m, tau_p), 101)
    n_samples = 1000

    l_mesh, t_mesh, _ = np.meshgrid(l, t, np.arange(n_samples), indexing='ij')

    g = np.zeros((l.size, t.size))

    n_runs = 10
    for _ in trange(n_runs):
        theta = (np.random.rand(l.size, t.size, n_samples) - 0.5) * np.pi

        t_at_ell_m = np.tan(theta) * (-ell - l_mesh) + t_mesh
        # t_at_ell_m = theta * (-ell - l_mesh) + t_mesh
        t_at_ell_p = np.tan(theta) * (ell - l_mesh) + t_mesh

        # detected = (t_at_ell_m >= -tau_m) & (t_at_ell_m <= tau_m)
        detected = (t_at_ell_m >= -tau_m) & (t_at_ell_m <= tau_m) & (t_at_ell_p >= -tau_p) & (t_at_ell_p <= tau_p)

        g += np.sum(detected, axis=-1)

    #
    g /= np.trapezoid(g, x=t, axis=1)[:, np.newaxis]

    l_plot = np.array([-ell, -ell / 2, 0, ell / 2, ell])
    idx_l, _ = np.where(l[:, np.newaxis] == l_plot[np.newaxis, :])

    std = np.trapezoid(g * t_mesh[:, :, 0] ** 2, x=t, axis=1)
    fig, ax = plt.subplots()
    ax.plot(l, np.sqrt(std * 6))
    ax.plot(l, np.sqrt(1 + (l / ell) ** 2) * (tau_m * (1 - l / ell) / 2 + tau_p * (1 + l / ell) / 2))
    ax.plot(l, np.sqrt(1 + (l / ell) ** 2) * tau_m)
    ax.plot(l, np.sqrt(1 + (l / ell) ** 2) * tau_p)
    ax.plot(l, np.sqrt((1 - l / ell) ** 2 * tau_m ** 2 + (1 + l / ell) ** 2 * tau_p ** 2) / np.sqrt(2))
    plt.show()

    # dd = np.zeros(l.size)
    # ee = np.zeros(l.size)
    #
    # for ii in range(l.size):
    #     abc = lambda a, tau, d: resolution_model(a, tau, tau, d, ell)
    #
    #     p_opt, p_cov = curve_fit(abc, t, g[ii, :], p0=[2.5, l[ii]])
    #
    #     dd[ii] = p_opt[0]
    #     ee[ii] = p_opt[1]
    #
    #     if ii in idx_l:
    #         fig, ax = plt.subplots()
    #         ax.plot(t, g[ii, :])
    #         ax.plot(t, abc(t, *p_opt))
    #         plt.show()
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(l, dd)
    # ax.plot(l, ee)
    # ax.plot(l, l)
    # plt.show()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots()
    ax.imshow(g.T, origin='lower')
    # for ii in range(l_plot.size):
    #     ax.plot(t, g[idx_l[ii], :], color=colors[ii])
    #     ax.plot(t, resolution_model(t, tau_m, tau_p, l_plot[ii], ell), linestyle='--', color=colors[ii])
    plt.show()


    return 0


def resolution_model(t, tau_m, tau_p, l, ell):
    #
    f = np.zeros(t.shape)

    tau = tau_m
    # tau = tau_m * (1 - l / ell) / 2 + tau_p * (1 + l / ell) / 2
    # print(tau)

    # Normalized coordinates
    t_n = np.abs(t) / tau
    l_n = np.abs(l) / ell

    plateau = t_n < l_n

    drop = (l_n <= t_n) & (t_n < 1)

    f[plateau] = 1
    f[drop] = (t_n[drop] - 1) / (l_n - 1)

    f /= tau * (l_n + 1)

    return f


if __name__ == '__main__':
    main()
