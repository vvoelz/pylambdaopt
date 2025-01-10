import os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# Analysis of the A16F EE results

def wl_increment(unopt_wl_files, opt_wl_files, outdir, ymin=1e-4):
    """Plot and analyze the average WL_increment over time for the trials of 
    unoptimized vs optimized A16F 

    INPUTS
    unopt_wl_files - a list of filenames ['wl_increment_trial1.txt',...] for each unopt trial
    opt_wl_files   - a list of filenames for the optimized trials
    outdir         - an output directory

    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ntrials_unopt = len(unopt_wl_files)
    ntrials_opt = len(opt_wl_files)

    plt.figure(figsize=(4,3.5))

    if ntrials_unopt > 0:
        # get the time values (in ns) from the first file on the list
        data = np.loadtxt(unopt_wl_files[0])
        time_in_ns = data[:,0]
        N = len(time_in_ns)

        wl_data = np.zeros( (N, ntrials_unopt) )
        for trial in range(ntrials_unopt):
            data = np.loadtxt(unopt_wl_files[trial])
            wl_data[:,trial] = data[:,1]

        wl_mean = wl_data.mean(axis=1)
        wl_std  = wl_data.std(axis=1)
        
        print('the average unopt WL_inc at 40 ns:', wl_mean[-1], 'kT')
        print('the average std(WL_inc) at 40 ns:', wl_std[-1], 'kT')

        plt.plot(time_in_ns, wl_mean, 'k-', label=r'before optimization ($K$=45)')
        plt.fill_between(time_in_ns, wl_mean+wl_std, wl_mean-wl_std, color='k', alpha=0.25, lw=0)

    if ntrials_opt > 0:
        # get the time values (in ns) from the first file on the list
        data = np.loadtxt(opt_wl_files[0])
        time_in_ns = data[:,0]
        N = len(time_in_ns)

        wl_data = np.zeros( (N, ntrials_opt) )
        for trial in range(ntrials_opt):
            data = np.loadtxt(opt_wl_files[trial])
            wl_data[:,trial] = data[:,1]

        wl_mean = wl_data.mean(axis=1)
        wl_std  = wl_data.std(axis=1)
    
        print('the average opt WL_inc at 40 ns:', wl_mean[-1], 'kT')
        print('the average std(WL_inc) at 40 ns:', wl_std[-1], 'kT')

        plt.plot(time_in_ns, wl_mean, 'r-', label=r'before optimization ($K$=26)')
        plt.fill_between(time_in_ns, wl_mean+wl_std, wl_mean-wl_std, color='r', alpha=0.25, lw=0)



    plt.xlabel('time (ns)')
    plt.ylabel(r'WL increment ($k_BT$)')

    plt.xlim(0,40.)
    plt.yscale('log')
    plt.ylim(ymin,20)

    # plt.legend()
    plt.tight_layout()

    outfile = os.path.join(outdir, 'WL_increment_over_time.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')



def dG(unopt_files, opt_files, outdir, ymin=1e-4):
    """Plot and analyze the average dG estimate over time for the trials of 
    unoptimized vs optimized A16F 

    INPUTS
    unopt_files - a list of filenames ['dG_trial1.txt',...] for each unopt trial
    opt_files   - a list of filenames for the optimized trials
    outdir         - an output directory

    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ntrials_unopt = len(unopt_files)
    ntrials_opt = len(opt_files)

    plt.figure(figsize=(4,3.5))

    if ntrials_unopt > 0:
        # get the time values (in ns) from the first file on the list
        data = np.loadtxt(unopt_files[0])
        time_in_ns = data[:,0]
        N = len(time_in_ns)

        dG_data = np.zeros( (N, ntrials_unopt) )
        for trial in range(ntrials_unopt):
            data = np.loadtxt(unopt_files[trial])
            dG_data[:,trial] = data[:,1]

        dG_mean = dG_data.mean(axis=1)
        dG_std  = dG_data.std(axis=1)

        print('the average dG at 40 ns:', dG_mean[-1], 'kT')
        print('the average std(dG) at 40 ns:', dG_std[-1], 'kT')
        

        plt.plot(time_in_ns, dG_mean, 'k-', label=r'before optimization ($K$=45)')
        plt.fill_between(time_in_ns, dG_mean+dG_std, dG_mean-dG_std, color='k', alpha=0.25, lw=0)

    if ntrials_opt > 0:
        # get the time values (in ns) from the first file on the list
        data = np.loadtxt(opt_files[0])
        time_in_ns = data[:,0]
        N = len(time_in_ns)

        dG_data = np.zeros( (N, ntrials_opt) )
        for trial in range(ntrials_opt):
            data = np.loadtxt(opt_files[trial])
            dG_data[:,trial] = data[:,1]

        dG_mean = dG_data.mean(axis=1)
        dG_std  = dG_data.std(axis=1)

        print('the average dG at 40 ns:', dG_mean[-1], 'kT')
        print('the average std(dG) at 40 ns:', dG_std[-1], 'kT')


        plt.plot(time_in_ns, dG_mean, 'r-', label=r'before optimization ($K$=26)')
        plt.fill_between(time_in_ns, dG_mean+dG_std, dG_mean-dG_std, color='r', alpha=0.25, lw=0)



    plt.xlabel('time (ns)')
    plt.ylabel(r'$\Delta G$ estimate ($k_BT$)')

    plt.xlim(0,40.)
    plt.ylim(-10,10)

    # plt.legend()

    plt.tight_layout()
    outfile = os.path.join(outdir, 'dG_over_time.pdf')
    plt.savefig(outfile)
    print(f'Wrote: {outfile}')







#######################

if __name__ == '__main__':


    import glob

    unopt_wl_files = glob.glob('unopt/wl_increment_trial*.txt')
    opt_wl_files = glob.glob('opt/wl_increment_trial*.txt')
    outdir = './'
    wl_increment(unopt_wl_files, opt_wl_files, outdir)

    unopt_dG_files = glob.glob('unopt/dG_trial*.txt')
    opt_dG_files = glob.glob('opt/dG_trial*.txt')
    outdir = './'
    dG(unopt_dG_files, opt_dG_files, outdir)



