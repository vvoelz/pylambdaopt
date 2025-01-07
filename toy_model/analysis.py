import os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec



# Analysis of the toy model EE results

def wl_increment(result_dir, outdir, ymin=1e-5):
    """Plot and analyze the WL_increment over time for the toy 1D harmonic potential model

    INPUTS
    result_dir - the directory containg the generated toy EE trajectories
    outdir     - a directory to save the analysis

    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    

    data = np.loadtxt(os.path.join(result_dir, 'wl_increment_traj.txt'))
    step_traj = data[:,0]
    k_traj    = data[:,1] 
    wl_increment_traj  = data[:,2]

    plt.figure(figsize=(4,3))
    plt.plot(step_traj, wl_increment_traj, 'r-')
    plt.xlabel('steps')
    plt.ylabel(r'WL increment ($k_BT$)')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin,20)

    plt.tight_layout()
    outfile = os.path.join(outdir, 'WL_increment.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')




def sampling_overview(result_dir, outdir, dG_limits=(-2,2), T_ticks=True, T_labels=True):
    """Plot an overview of the potentials, transition probs, k trajectory, and dG trajectory."""

    if not os.path.exists(outdir):
        os.mkdir(outdir)


    # load in the data to plot

    lambdas = np.loadtxt(os.path.join(result_dir,'lambda_values.txt'))
    K = len(lambdas)  # number of intermediates

    T1 =  np.loadtxt(os.path.join(result_dir,'transition_matrix.txt'))

    data = np.loadtxt(os.path.join(result_dir, 'f_k_traj.txt'))
    step_traj = data[:,0]
    k_traj    = data[:,1]
    f_k_traj  = data[:,2:]
    dG_traj   = f_k_traj[:,-1] - f_k_traj[:,0]



    # plt.rc('axes', labelsize=18,titlesize=18)   # For xlabel and ylabel
    # plt.rc('xtick', labelsize=18)  # For x-axis tick labels
    # plt.rc('ytick', labelsize=18) 
    xvalues = np.arange(-2., 11, 0.05)

    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(2, 4, figure=fig,height_ratios=[1,2])  # 2 rows, 8 columns

    # Top row: Two long plots
    ax1 = fig.add_subplot(gs[0, :])  # First long plot (left)

    def u(x, x0, k_f=10.0):
        return k_f/2.0 * (x - x0)**2 

    for i in range(K):
        x0 = 10.0*lambdas[i]
        ax1.plot(xvalues, u(xvalues, x0), label='i=%d'%i)
    ax1.set_ylim(0,5)
    ax1.set_yticks([0,2,4])
    ax1.set_xlim(-1,11)
    # ax1.set_title("Thermo. Ensembles Unopt.")

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$U_k(x)$ (kT)')


    # Bottom row: Four square plots under each long plot

    l1, l2, l3, l4 = [fig.add_subplot(gs[1, j]) for j in range(4)]

    l1.pcolor(T1, cmap='gray_r', vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if abs(i-j) < 2:
                c = 'w'
                if T1[i, j] != 0:  # Check for non-zero cell
                    c = 'k'
                if T_labels:
                    l1.text(j + 0.5, i + 0.5, f"{T1[i, j]:.2f}", ha='center', va='center', color=c, fontsize=5)

    # l1: Transition probabilities
    if T_ticks:
        l1.set_xticks(np.arange(0.5, K+0.5,1))
        l1.set_xticklabels(np.arange(1,K+1,1))
    if T_ticks:
        l1.set_yticks(np.arange(0.5, K+0.5,1))
        l1.set_yticklabels(np.arange(1,K+1,1))
    l1.set_xlabel(r'$k$')
    l1.set_ylabel(r'$k$')

    # l2: k trajectory
    l2.plot(step_traj, k_traj+1, c='b', lw=0.5)
    l2.set_xlabel('steps')
    l2.set_ylabel(r'$k$')
    l2.set_xscale('log')
    l2.set_xlim(10**4,10**7)

    # l3: Delta f over full range
    l3.plot(step_traj, dG_traj, c='b', lw=1)
    #l3.set_title(r"Unopt. $\Delta G$ Traj.")
    plot_x = np.linspace(0,1e7,len(dG_traj))
    l3.fill_between(plot_x,-np.ones_like(plot_x),np.ones_like(plot_x), color="k", alpha=0.1)
    l3.hlines(0,0,1e7,color='r')
    l3.set_xlabel('steps')
    l3.set_ylabel(r'$\Delta \tilde{f}$ (kT)')
    #ax4.set_yticks([-10,-5,0,5,10])
    #l3.set_ylim(dG_limits)
    l3.set_xlim(1e4,1e7)
    l3.set_xscale('log')

    # l4: Delta f over dG limits, near zero
    l4.plot(step_traj, dG_traj, c='b', lw=1)
    plot_x = np.linspace(0,1e7,len(dG_traj))
    l4.fill_between(plot_x,-np.ones_like(plot_x),np.ones_like(plot_x), color="k", alpha=0.1)
    l4.hlines(0,0,1e7,color='r')
    l4.set_xlabel('steps')
    # l4.set_ylabel(r'$\Delta \tilde{f}$ (kT)')
    l4.set_ylim(dG_limits)
    l4.set_xlim(1e4,1e7)
    l4.set_xscale('log')

    plt.tight_layout()
    outfile = os.path.join(outdir, 'sampling_overview.pdf') 
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')


#######################

if __name__ == '__main__':


    # run tests of the analysis routines above

    #wl_increment('testout', 'analysis_testout')
    #sampling_overview('testout', 'analysis_testout')

    #wl_increment('testout_optimized', 'analysis_testout_optimized')
    #sampling_overview('testout_optimized', 'analysis_testout_optimized')

    if (0):
        wl_increment('test10M_unoptimized', 'analysis_test10M_unoptimized')
        sampling_overview('test10M_unoptimized', 'analysis_test10M_unoptimized')

    if (0):
        wl_increment('test10M_optimized', 'analysis_test10M_optimized')
        sampling_overview('test10M_optimized', 'analysis_test10M_optimized')

    if (1): 
        for trial in range(5):
            sampling_overview(f'test10M_K25_uniform_trial{trial}', f'analysis_test10M_K25_uniform_trial{trial}', T_ticks=False, T_labels=False)

    #wl_increment('test10M_K50_optimized', 'analysis_test10M_K50_optimized')
    #sampling_overview('test10M_K50_optimized', 'analysis_test10M_K50_optimized')

