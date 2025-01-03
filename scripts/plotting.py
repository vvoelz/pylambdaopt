import os, sys
import numpy as np
from matplotlib import pyplot as plt



# Here we define several plotting routines to visualize the optimization process and results

def plot_spline(alpha_values, L_values, L_spl, L_spl_1d, filename, d_alpha=0.01):
    """Makes a two-panel plot of the cubic spline and its derivative.

    INPUTS
    alpha_values  - the list of alpha_k values
    L_values      - the list of observed L values
    L_spl         - the cubic spline function (from Optimizxer.L_spl )
    L_spl_1d      - the 1st derivative of the cubic spline function (from Optimizxer.L_spl_1d )
    filename      - the filename to write the plot  

    PARAMETERS
    d_alpha       - resolution to compute and plot the continuous L(alpha) curve
    """

    plt.figure(figsize=(6,3))

    # plot the cubic spline in the first panel
    if (1):
        plt.subplot(1,2,1)
        plt.plot(alpha_values, L_values, 'ro', label = 'data')

        alpha_range = np.arange(alpha_values[0], alpha_values[-1], d_alpha)
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


 

