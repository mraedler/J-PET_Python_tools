"""
Data visualization

@author: Martin Rädler
"""
# Python libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    tau_1 = 2
    tau_2 = 3
    ell = 20

    # t = np.linspace(-2 * tau_2, 2 * tau_2, 101)
    t = np.linspace(-tau_2 - 1, tau_2 + 1, 101)
    t = np.linspace(-2 * tau_2, 2 * tau_2, 101)
    l = np.linspace(-ell, ell, 201)

    dt, dl = t[1] - t[0], l[1] - l[0]

    # extent = (t[0] - dt / 2, t[-1] + dt / 2, l[0] - dl / 2, l[-1] + dl / 2)
    extent = (l[0] - dl / 2, l[-1] + dl / 2, t[0] - dt / 2, t[-1] + dt / 2)

    # t_mesh, l_mesh = np.meshgrid(t, l, indexing='ij')
    l_mesh, t_mesh = np.meshgrid(l, t, indexing='ij')

    # f = resolution_model(t_mesh, tau, l_mesh, ell)
    f = resolution_model_2(t_mesh, l_mesh, tau_1, tau_1, ell)
    # f = resolution_model_2(t_mesh, l_mesh, tau_1, tau_2, ell)
    # f = resolution_model_2(t_mesh, l_mesh, tau_2, tau_2, ell)

    # fig, ax = plt.subplots()
    # # ax.plot(np.trapezoid(f, x=t, axis=1))
    # ax.plot(np.trapezoid(f, x=l, axis=0))
    # plt.show()

    x_ticks = [-ell, -ell / 2, 0, ell / 2, ell]

    plt.rcParams.update({'font.size': 16})
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    ax1.imshow(f.T, origin='lower', cmap='gray_r', extent=extent, alpha=0.5)
    ax1.plot([-ell, ell], [0, 0], color='k')
    ax1.plot([0, 0], [-tau_1, tau_1], color='k')
    ax1.set_aspect('auto')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([r'$-\ell$', r'$-\ell/2$', r'$0$', r'$\ell/2$', r'$\ell$'])
    ax1.set_ylim([-7, 7])
    ax1.set_yticks([-tau_1, 0, tau_1])
    ax1.set_yticklabels([r'$-\tau_1$', r'$0$', r'$\tau_1$'])
    # ax1.set_yticks([-tau_2, 0, tau_2])
    # ax1.set_yticklabels([r'$-\tau_2$', r'$0$', r'$\tau_2$'])

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(ax1.get_ylim())
    ax1_twin.set_yticks([-tau_1, 0, tau_1])
    ax1_twin.set_yticklabels([r'$-\tau_1$', r'$0$', r'$\tau_1$'])
    # ax1_twin.set_yticks([-tau_2, 0, tau_2])
    # ax1_twin.set_yticklabels([r'$-\tau_2$', r'$0$', r'$\tau_2$'])

    for spine in ax1.spines.values():
        spine.set_visible(False)

    for spine in ax1_twin.spines.values():
        spine.set_visible(False)

    ax1.set_xlabel(r'$l$, $t$, D1, D2')
    # ax1.set_ylabel(r'$t$')

    for ll in x_ticks:
        ax0.plot(np.linspace(-2, 2, 101) + ll, f[l == ll, :].squeeze(), color='k')
    ax0.set_ylim([0, 2.5])
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_xlim([-24.2, 24.2])

    for spine in ax0.spines.values():
        spine.set_visible(False)

    ax1.set_xlim(ax0.get_xlim())

    plt.show()

    return 0


def resolution_model(t, tau, l, ell):
    #
    f = np.zeros(t.shape)

    # Normalized coordinates
    t_n = np.abs(t) / tau
    l_n = np.abs(l) / ell

    plateau = t_n < l_n

    drop = (l_n <= t_n) & (t_n < 1)

    f[plateau] = 1
    f[drop] = (t_n[drop] - 1) / (l_n[drop] - 1)

    f /= tau * (l_n + 1)

    return f


def resolution_model_2(t, l, tau_1, tau_2, ell):
    # Allocate
    g = np.zeros(t.shape)

    #
    tau_p = tau_1 * (1 - l / ell) / 2 + tau_2 * (1 + l / ell) / 2
    tau_m = tau_1 * (1 - l / ell) / 2 - tau_2 * (1 + l / ell) / 2

    # tau_m appears only in absolute value
    tau_m = np.abs(tau_m)

    plateau = tau_m > np.abs(t)
    drop = (tau_m <= np.abs(t)) & (np.abs(t) < tau_p)

    g[plateau] = tau_p[plateau] - tau_m[plateau]
    g[drop] = tau_p[drop] - np.abs(t[drop])

    #
    l_1 = l / ell == -1.
    l_2 = l / ell == 1.
    l_valid = (~l_1) & (~l_2)
    g[l_valid] /= tau_1 * tau_2 * (1 - l[l_valid] ** 2 / ell ** 2)

    def heaviside(x): return np.heaviside(x, 1.)

    g[l_1] = (heaviside(t[l_1] + tau_1) - heaviside(t[l_1] - tau_1)) / (2 * tau_1)
    g[l_2] = (heaviside(t[l_2] + tau_2) - heaviside(t[l_2] - tau_2)) / (2 * tau_2)

    return g


if __name__ == "__main__":
    main()
