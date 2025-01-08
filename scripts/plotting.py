import os, sys
import numpy as np
from matplotlib import pyplot as plt



# Here we define several plotting routines to visualize the optimization process and results

def plot_spline(alpha_values, L_values, L_spl, L_spl_1d, filename):
    """Makes a two-panel plot of the cubic spline and its derivative.

    INPUTS
    alpha_values  - the list of alpha_k values
    L_values      - the list of observed L values
    L_spl         - the cubic spline function (from Optimizxer.L_spl )
    L_spl_1d      - the 1st derivative of the cubic spline function (from Optimizxer.L_spl_1d )
    filename      - the filename to write the plot  

    """

    plt.figure(figsize=(6,3))

    # plot the cubic spline in the first panel
    if (1):
        plt.subplot(1,2,1)
        plt.plot(alpha_values, L_values, 'ro', label = 'data', ms=4)

        alpha_range = np.linspace(alpha_values[0], alpha_values[-1], 1000)
        plt.plot(alpha_range, L_spl(alpha_range), label="spline")   # for UnivariateSpline
        ## plt.plot(x_observed, y_spl(x_observed), label="spline") # for CubicSpline

        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mathcal{L}(\alpha)$')

    # plot the derivative of the cubic spline in the second panel
    if (1):
        plt.subplot(1,2, 2)   #derivative plot
        plt.plot(alpha_values, L_spl_1d(alpha_values), '.', label='data')
        plt.plot(alpha_range, L_spl_1d(alpha_range), '-', label='spline')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$d\mathcal{L}/d\alpha$')

    plt.tight_layout()
    plt.savefig(filename)
    print(f'Wrote: {filename}')


def plot_opt_traces(traj_alphas, L_spl, filename):
    """Makes a two-panel plot of the optimization trajectory:
    traces of the alpha values vs optimization step, and 
    traces of the L(alpha) values vs optimization step.""" 


    K, nsteps = traj_alphas.shape

    plt.figure(figsize=(6,3))

    # Plot alpha values vs optimization step
    plt.subplot(1,2,1)
    for i in range(K):
        plt.plot(np.arange(nsteps), traj_alphas[i,0:nsteps], '-')
    plt.xlabel('step')
    plt.ylabel(r'$\alpha_k$ values')
    
    plt.subplot(1,2,2)
    for i in range(K):
        plt.plot(np.arange(nsteps), L_spl(traj_alphas[i,0:nsteps]), '-')
    plt.xlabel('step')
    plt.ylabel(r'$\mathcal{L}(\alpha_k)$ values')

 
    plt.tight_layout()
    plt.savefig(filename)
    print(f'Wrote: {filename}')


def plot_old_vs_new_alphas(old_alphas, new_alphas, L_spl, filename):
    """makes a two panel plot of thee old and new alpha values marked at
    locations on the L(alpha) vs alpha plot."""

    plt.figure(figsize=(8,4))

    alpha_range = np.linspace(old_alphas[0], old_alphas[-1], 1000)

    if (1):
        # Plot the old alphas on the L(alphas) vs alpha plot
        plt.subplot(2,1,1)
        plt.plot(alpha_range, L_spl(alpha_range), 'b-', label="spline")
        plt.plot(old_alphas, L_spl(np.array(old_alphas)), 'r.', label="old alphas")
        for value in old_alphas:
            plt.plot([value, value], [0, L_spl(value)], 'r-')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mathcal{L}(\alpha)$')
        plt.title(r'old $\alpha_k$ values')

    if (1):
        # Plot the new alphas on the L(alphas) vs alpha plot
        plt.subplot(2,1,2)
        plt.plot(alpha_range, L_spl(alpha_range), 'b-', label="spline")
        plt.plot(new_alphas, L_spl(new_alphas), 'g.', label="new alphas")
        for value in new_alphas:
            plt.plot([value, value], [0, L_spl(value)], 'g-')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mathcal{L}(\alpha)$')
        plt.title(r'new $\alpha_k$ values')

    plt.tight_layout()
    plt.savefig(filename)
    print(f'Wrote: {filename}')


def plot_mixing(K_values, P_acc_values, t2_values, filename, max_t2=1e7):
    """Makes a plot of the mixing time t2 vs the number of alchemical intermediates K."""


    plt.figure(figsize=(5,4))

    Ind = np.isfinite(t2_values)*(np.array(t2_values) < max_t2)
    finite_K_values = np.array(K_values)[Ind]
    finite_t2_values = np.array(t2_values)[Ind]
    plt.plot(finite_K_values, finite_t2_values, 'b-', label=r'$t_2(K)$')

    # find the minimum and mark it
    Ind = np.argmin(finite_t2_values)
    plt.plot(finite_K_values[Ind], finite_t2_values[Ind], 'r*', label=f'minimum at K={finite_K_values[Ind]}')

    plt.yscale('log')
    plt.xlabel(r'number of intermediates $K$')
    plt.ylabel(r'mixing time $t_2$ (in units $\tau$)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f'Wrote: {filename}')



