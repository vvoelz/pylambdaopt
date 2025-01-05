import os, sys
import numpy as np
import matplotlib.pyplot as plt

#  %matplotlib inline

import scipy 
from scipy import stats	

sys.path.append('../scripts')

from optimizer import *
from plotting import *


def optimize_toy_lambdas(result_dir, outdir, make_plots=True, save_plots=True, verbose=True):
    """Optimize the lambda values for the toy 1D harmonic potential model

    INPUTS
    result_dir - the directory containg the generated toy EE trajectories
    outdir     - a directory to save the analysis

    PARAMETERS
    make_plots - will generate plots if True
    save_plots - will save the generated plots to file if True
    verbose    - will print more text to stanard output reporting status if True

    RETURNS
        fep-lambdas (their optimized values)

    """

    u_k_data = np.loadtxt(os.path.join(result_dir, 'u_k_traj.txt'))
    step_traj = u_k_data[:,0]
    k_traj    = u_k_data[:,1] 
    u_k_traj  = u_k_data[:,2:]
   

    # For a simple FEP alchemical transformation, there is only one lambda parameter,
    # so that phi(\alpha_k) = \lambda_k

    lambdas = np.loadtxt(os.path.join(result_dir,'lambda_values.txt'))
    K = len(lambdas)  # number of intermediates

    # estimate the sigmas from the u_k data
    sigmas = []
    for k in range(K-1) :

        Ind = np.where( k_traj == k ) 
        print('Ind', Ind)
        fwd_work_values = u_k_traj[Ind,k+1] - u_k_traj[Ind,k]
        fwd_work_values = fwd_work_values[ int(len(Ind)/2): ]   # take last half
        print('fwd_work_values', fwd_work_values)
        sigma_fwd  = np.std(fwd_work_values) # std dev of forward instantanous work values

        Ind2 = np.where( k_traj == (k+1) ) 
        print('Ind2', Ind2)
        bwd_work_values = u_k_traj[Ind2,k] - u_k_traj[Ind2,k+1]
        bwd_work_values = bwd_work_values[ int(len(Ind)/2): ]   # take last half
        sigma_bwd = np.std(bwd_work_values)  # std dev of forward instantanous work values

        print('k =',k, 'sigma_fwd', sigma_fwd, 'sigma_bwd', sigma_bwd)
        sigmas.append( (sigma_fwd + sigma_bwd)/2.0 )
        
    print('sigmas', sigmas) 


    #################################################
    ### alchemical optimization

    alpha_values = lambdas	

    L_values = np.cumsum(sigmas)    # convert to a list of L(0,\alpha_k) values
    L_values = np.array(np.concatenate([[0], L_values]))    # add a zero corresponding to \alpha[0] = 0.0
    ## VAV: This zero needs to be included.  Why was this left out before?

    print('L_values', L_values)

    # Create an Optimizer() object
    o = Optimizer()

    # create the cubic spline and 1st deriv functions, o.L_spl() and o.L_spl_1d()
    o.create_spline(alpha_values, L_values)
    print('o.L_spl', o.L_spl, 'o.L_spl_1d', o.L_spl_1d)

    if make_plots:
        # make plots of the spline
        spline_pngfile = os.path.join(outdir, f'splinefit.png') 
        spline_pdffile = os.path.join(outdir, f'splinefit.pdf')
        plot_spline(alpha_values, L_values, o.L_spl, o.L_spl_1d, spline_pdffile)


    # Optimize the \alpha_k values
    new_alphas, traj_alphas = o.optimize_alphas(alpha_values)

   
    if make_plots:     
        # make plots of the optimization traces
        traces_pngfile = os.path.join(outdir, f'opt_traces.png')
        traces_pdffile = os.path.join(outdir, f'opt_traces.pdf')
        plot_opt_traces(traj_alphas, o.L_spl, traces_pdffile)

    
    if make_plots:
        # make plots of old versus new alphas
        old_new_pngfile = os.path.join(outdir, f'old_vs_new_alphas.png')
        old_new_pdffile = os.path.join(outdir, f'old_vs_new_alphas.pdf')
        plot_old_vs_new_alphas(alpha_values, new_alphas, o.L_spl, old_new_pdffile)


    # create a path function $\phi(\alpha) \rightarrow \vec{\lambda} to map the alphas back to lambdas
    phi = o.create_phi(labels=['FEP'], ascending=[True])

    print('new_alphas', new_alphas)

    # perform the mapping and return the new lambdas
    new_lambdas = phi(new_alphas)

    return new_lambdas




########################

if __name__ == '__main__':

    usage  = """Usage:    python optimize_toy_lambdas.py result_dir outdir

    OUTPUT
        * Numpy arrays of the new values will be written to:
            - [outdir]/new_lambdas.txt
        * graphs associated with the lambda optimization will be written as images:
            - [outdir]/optimization_traces.pdf
            - [outdir]/old_vs_new_lambdas.pdf
            - [outdir]/splinefit.pdf
              
    EXAMPLE
    Try this:
        $ python optimize_toy_lambdas.py testout testopt 
    """        

    # Parse input

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    result_dir    = sys.argv[1]
    outdir       = sys.argv[2]

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    assert os.path.exists(outdir)

    # Optimize the lambdas
    new_toy_lambdas = optimize_toy_lambdas(result_dir, outdir)


    # print out new fep_lambdass to std output, formatted like an mdp file
    print('### Optimized lambda values ###')
    print('')
    print('new_toy_lambdas', new_toy_lambdas)


