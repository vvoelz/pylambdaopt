import os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec



# Analysis of the toy model EE results

def wl_increment_vs_K(result_dir_list, K_values, outdir, ymin=1e-5):
    """Plot and analyze the WL_increment over time for the toy 1D harmonic potential model

    INPUTS
    result_dirs - a list of directories containing the generated toy EE trajectories
    K_values    - a correpsponding list of K values
    outdir     - a directory to save the analysis

    """

    assert len(result_dir_list) == len(K_values)  

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    
    plt.figure(figsize=(4,3))

    for i in range(len(result_dir_list)):

        result_dir = result_dir_list[i]
        K = K_values[i]

        data = np.loadtxt(os.path.join(result_dir, 'wl_increment_traj.txt'))
        step_traj = data[:,0]
        k_traj    = data[:,1]
        wl_increment_traj  = data[:,2]

        plt.plot(step_traj, wl_increment_traj, label=f'K={K}')

    plt.xlabel('steps')
    plt.ylabel(r'WL increment ($k_BT$)')
    plt.legend(loc='best')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin,20)

    plt.tight_layout()
    outfile = os.path.join(outdir, 'WL_convergence_vs_K.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')


def wl_increment_unopt_vs_opt(result_dir_unopt, result_dir_opt, outdir, ymin=1e-5):
    """Plot and analyze the WL_increment over time for the toy 1D harmonic potential model

    INPUTS
    result_dirs - a list of lists of directories (!) containing the generated toy EE trajectories
    K_values    - a correpsponding list of K values
    outdir     - a directory to save the analysis

    """


    if not os.path.exists(outdir):
        os.mkdir(outdir)


    plt.figure(figsize=(4,3))

    result_dir_list = [result_dir_unopt, result_dir_opt]
    labels = ['before optimization', 'after optimization']

    for i in range(len(result_dir_list)):

        result_dir = result_dir_list[i]

        data = np.loadtxt(os.path.join(result_dir, 'wl_increment_traj.txt'))
        step_traj = data[:,0]
        k_traj    = data[:,1]
        wl_increment_traj  = data[:,2]

        plt.plot(step_traj, wl_increment_traj, label=labels[i])

    plt.xlabel('steps')
    plt.ylabel(r'WL increment ($k_BT$)')
    plt.legend(loc='best')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin,20)

    plt.tight_layout()
    outfile = os.path.join(outdir, 'WL_convergence_unopt_vs_opt.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')



def dG_vs_K(result_dir_list, K_values, outdir, last_snaps=20000, wl_thresh=0.001):
    """Plot and analyze the final estimates of dG as an average over the last few steps

    INPUTS
    result_dirs - a list of LISTS of directories containg the generated toy EE trajectories
    K_values    - a correpsponding list of K values
    outdir     - a directory to save the analysis

    PARAMETERS
    last_snaps - the last snapshots in the f_k_traj.txt file to compute the averages

    """

    assert len(result_dir_list) == len(K_values)  

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dG_means = []    # Lsts for each K value
    dG_stds = []

    for i in range(len(result_dir_list)):

        K = K_values[i]


        dG_means_for_thisK = []

        result_dirs = result_dir_list[i]
        for result_dir in result_dirs:
                
            data = np.loadtxt(os.path.join(result_dir, 'f_k_traj.txt'))
            step_traj = data[:,0]
            k_traj    = data[:,1]
            f_k_traj  = data[:,2:]
            dG_traj   = f_k_traj[:,-1] - f_k_traj[:,0]

            wl_data = np.loadtxt(os.path.join(result_dir, 'wl_increment_traj.txt'))
            wl_increment_traj  = wl_data[:,2]


            if (1):
                # collect dG samples after WL thresh is reached
                Ind = np.where(wl_increment_traj < wl_thresh)
                dG_means_for_thisK.append( np.sqrt( np.mean( (dG_traj[Ind])**2  )) )
            else:
                dG_means_for_thisK.append( np.sqrt( np.mean( (dG_traj[-last_snaps:])**2  )) )

        dG_means.append( np.mean( dG_means_for_thisK ))
        dG_stds.append( np.std( dG_means_for_thisK ))

    plt.figure(figsize=(4,3))
    plt.errorbar(K_values, dG_means, yerr=dG_stds)
    plt.plot(K_values, dG_means,  'ko', ms=4)
    plt.xlabel(r'number of intermediates $K$')
    plt.ylabel(r'error in $\Delta \tilde{f}$ ($k_BT$)')


    plt.tight_layout()
    outfile = os.path.join(outdir, 'dG_vs_K.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')






#######################

if __name__ == '__main__':


    K_values = list(range(10,100,5))

    result_dirs = []
    for K in K_values:
        result_dirs.append( [f'test10M_K{K}_uniform_trial{i}' for i in range(5)] )

    if (1):
        wl_increment_unopt_vs_opt('test10M_unoptimized', 'test10M_optimized', 'analysis_test10M_optimized')
    else:
        dG_vs_K(result_dirs, K_values, 'analysis_test10M_multiK_optimized')

