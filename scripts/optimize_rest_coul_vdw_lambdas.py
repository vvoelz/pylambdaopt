import os, sys
import numpy as np
import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	

from ee_tools import *


def optimize_rest_coul_vdw_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, make_plots=True, save_plots=True, verbose=True):
    """
    DESCRIPTION
        Optimize the lambda values for all intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    RETURNS
        new_rest_lambdas, new_coul_lambdas, new_vdw_lambdas   (their optimized values)

    WARNING
        This script *assumes* that the rest, coul and vdw lambda values
        are changed sequentially, in that order!  i.e. First the rest is turned on from 0.0 to 1.0, then electrostates are turned off, 
        with coul lambda going from 0.0 to 1.0, then the vdw are turned off, with lambda going from 0.0 to 1.0. 
        Ideally, one intermediates has coul-lambda = 1.0, vdw-lambda = 0.0. 
    """


    rest_lambdas, coul_lambdas, vdw_lambdas = get_rest_coul_vdw_lambdas(mdpfile)

    # We map the three sets of values \in [0,1] to interval [0,3]
    lambdas = rest_lambdas + coul_lambdas + vdw_lambdas
    if verbose:
        print('lambdas', lambdas)
        print('lambdas.shape', lambdas.shape)

    time_in_ps, thermo_states, dhdl, pV = get_dhdl_data(dhdl_xvgfile)
    if verbose:
        print('time_in_ps', time_in_ps)
        print('thermo_states', thermo_states)
        print('dhdl', dhdl)
        print('pV', pV)

    sigmas = estimate_sigmas(dhdl, thermo_states, plot_data=False)
    if verbose:
        print('sigmas.shape', sigmas.shape)
 
    ### HOT FIX
    ### If any of the sigma values are zero, then we know this is one of the cases where there are duplicate
    ### fep-lambda

    print('sigmas', sigmas)
    indices_for_which_sigma_is_zero = np.where(sigmas < 0.0001)[0]
    print('indices_for_which_sigma_is_zero', indices_for_which_sigma_is_zero)

    # There should be only one such duplicate!
    if len(indices_for_which_sigma_is_zero) == 1:
        # For *THIS* particular data set we can see there is a problem:
        # The i=49 --> i+1 = 50 lambdas are the same, resulting in a sigma of zero for that transition
        # Let's remove it
     
        remove_index = indices_for_which_sigma_is_zero[0]
        print('Removing DUPLICATE lambda index:', remove_index)
        # print('sigmas', sigmas)
        sigmas_list = sigmas.tolist()
        sigmas_list.pop(remove_index)
        sigmas = np.array(sigmas_list)
        print('FIXED sigmas', sigmas)

        print('lambdas',lambdas)
        print('lambdas.shape',lambdas.shape)
        lambdas_list = lambdas.tolist()
        lambdas_list.pop(remove_index)
        lambdas = np.array(lambdas_list)
        print('FIXED lambdas',lambdas)
        print('FIXED lambdas.shape',lambdas.shape)

    elif len(indices_for_which_sigma_is_zero) > 1:
        raise Exception("There are multiple zero sigma! Are there multiple duplicates in the lambda values?!")

    else:
        print('There are no duplicate lambda')
        pass


    ### Lambda optimization

    dx = sigmas                 #according to Vince's equation (VAV: k is set to 1)
    
    x_values = np.cumsum(dx)    # convert to a list of x values of separated harmonic potentials
    x_values = np.array(np.concatenate([[0], x_values]))    # add a zero corresponding to lambda0 = 0.0
    ## VAV: This zero needs to be included.  Why was this left out before?
    print('x_values', x_values)


    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import interp1d

    if make_plots:
        plt.figure(figsize=(12,6))

    lambda_values = lambdas #not inclduing the first one, lambda_0 

    x_observed = lambda_values      #not inclduing the first one, lambda_0
    y_observed = x_values

    if make_plots:
        plt.subplot(1,2,1)
        plt.plot(x_observed, y_observed, 'ro', label = 'data')
        #plt.semilogy(x_observed, y_observed, 'ro', label = 'data')

    #y_spl = CubicSpline(x_observed, y_observed)#, s=0,k=4)  
    y_spl = UnivariateSpline(x_observed, y_observed, s=0, k=3)  
    x_range = np.linspace(x_observed[0], x_observed[-1], 1000)


    if make_plots:
        plt.plot(x_range, y_spl(x_range), label="spline")   # for UnivariateSpline
        ## plt.plot(x_observed, y_spl(x_observed), label="spline") # for CubicSpline
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')

        plt.subplot(1,2, 2)   #derivative plot

    y_spl_1d = y_spl.derivative(n=1)    #n=1 , means the first order derivative
    #print (y_spl_1d(x_observed))
    # y_spl_1d = y_spl(x_observed, 1)  # first derivative of Cubic spline

    if make_plots:
        plt.plot(x_range, y_spl_1d(x_range), '-')
        plt.plot(x_observed, y_spl_1d(x_observed), '.')
        plt.ylabel('dx/dlambda')
        
        #plt.plot(x_observed, y_spl_1d, '.-', label='derivative')
        plt.legend()
        plt.xlabel('lambda')

        if save_plots:
            spline_pngfile = os.path.join(outdir, f'{outname}_splinefit.png') 
            plt.savefig(spline_pngfile)
            print(f'Wrote: {spline_pngfile}')



    # Let's try a steepest descent algorithm like the kind I wrote up in "math-gradient-descent-2021-05-07.pdf" -VAV

    # run the algorithm some fixed number of steps, or until some tolerance is reached
    nsteps = 100000
    tol = 1e-7  # stop if the lambdas dont change within this tolerance

    alpha = 1e-5  # gradient descent step size
    max_del_lambda = 0.0001   # the minimization step limited to this as a maximum change

    print_every = 2000
    
    nlambdas = len(lambda_values)
    print('lambda_values', lambda_values)
    old_lambdas = np.array(lambda_values)
    traj_lambdas = np.zeros( (nlambdas,nsteps) )
    for step in range(nsteps):

        # store the trajectory of lambdas
        traj_lambdas[:,step] = old_lambdas
        if verbose:
            print('step', step, old_lambdas)
    
        # perform a steepest descent step
        new_lambdas = np.zeros( old_lambdas.shape )
        del_lambdas = np.zeros( old_lambdas.shape )
        del_lambdas[0] = 0.0   # fix the \lambda = 0 endpoint
        del_lambdas[nlambdas-1] = 0.0  # fix the \lambda = 1 endpoint
    
        if False:  # do in a loop (SLOW!) 
            for i in range(1, (nlambdas-1)):
                del_lambdas[i] = -1.0*alpha*2.0*y_spl_1d(old_lambdas[i])*( 2.0*y_spl(old_lambdas[i]) - y_spl(old_lambdas[i-1]) - y_spl(old_lambdas[i+1]))
        else:   # do as a vector operation (FAST!) 
            y_all = y_spl(old_lambdas)
            yh, yi, yj = y_all[0:nlambdas-2], y_all[1:nlambdas-1], y_all[2:nlambdas] 
            del_lambdas[1:nlambdas-1] = -1.0*alpha*2.0*y_spl_1d(old_lambdas[1:nlambdas-1])*( 2.0*yi - yh - yj)
        if abs(np.max(del_lambdas)) > max_del_lambda:
            del_lambdas[1:nlambdas-1] = del_lambdas[1:nlambdas-1]*max_del_lambda/np.max(del_lambdas)
        new_lambdas = old_lambdas + del_lambdas
        
        # record the average change in the lambdas 
        del_lambdas = np.abs(old_lambdas - new_lambdas).mean()
        if step % print_every == 0:
            print('step', step, 'del_lambdas', del_lambdas)
        if del_lambdas < tol:
            print('Tolerance has been reached: del_lambdas =', del_lambdas, '< tol =', tol)
            break
        
        old_lambdas = new_lambdas
   
    if make_plots:     
    
        # Plot the results
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        for i in range(nlambdas):
            plt.plot(range(step), traj_lambdas[i,0:step], '-')
        plt.xlabel('step')
        plt.ylabel('lambda values')
        
        plt.subplot(1,2,2)
        for i in range(nlambdas):
            plt.plot(range(step), y_spl(traj_lambdas[i,0:step]), '-')
        plt.xlabel('step')
        plt.ylabel('x values')

        if save_plots:
            traces_pngfile = os.path.join(outdir, f'{outname}_optimization_traces.png')
            plt.savefig(traces_pngfile)
            print(f'Wrote: {traces_pngfile}')


    if make_plots:

        plt.figure(figsize=(12,4))

        plt.subplot(2,1,1)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(lambda_values, y_spl(np.array(lambda_values)), 'r.', label="old lambdas")
        for value in lambda_values:
            plt.plot([value, value], [0, y_spl(value)], 'r-')
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')
        plt.title('old lambdas')

        plt.subplot(2,1,2)
        plt.plot(x_range, y_spl(x_range), 'b-', label="spline")
        plt.plot(new_lambdas, y_spl(new_lambdas), 'g.', label="new lambdas")
        for value in new_lambdas:
            plt.plot([value, value], [0, y_spl(value)], 'g-')
        plt.legend()
        plt.xlabel('lambda')
        plt.ylabel('x values')
        plt.title('new lambdas')

        plt.tight_layout()

        if save_plots:
            old_vs_new_lambdas_pngfile = os.path.join(outdir, f'{outname}_old_vs_new_lambdas.png')
            plt.savefig(old_vs_new_lambdas_pngfile)
            print(f'Wrote: {old_vs_new_lambdas_pngfile}')

    # map the sum of coul-lambdas and vdw-lambdas back to their [0,1] ranges
    new_rest_lambdas = np.minimum(new_lambdas, 1.0) 
    new_coul_lambdas = np.maximum(np.minimum(new_lambdas, 2.0), 1.0)   - 1.0
    new_vdw_lambdas  = np.maximum(new_lambdas, 2.0) - 2.0

    return new_rest_lambdas, new_coul_lambdas, new_vdw_lambdas


########################

if __name__ == '__main__':

    usage  = """Usage:    python optimize_rest_coul_vdw__lambdas.py mdpfile dhdl_xvgfile outname outdir

    DESCRIPTION
        This script will optimize the lambda values for all intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    OUTPUT
        * A mdpfile-compatible string with new coul-lambdas and vdw-lambdas printed to std output
        * Numpy arrays of the new values will be written to:
            - [outdir]/[outname]_new_rest_lambdas.py
            - [outdir]/[outname]_new_coul_lambdas.py
            - [outdir]/[outname]_new_vdw_lambdas.py
        * graphs associated with the lambda optimization will be written as images:
            - [outdir]/[outname]_optimization_traces.png
            - [outdir]/[outname]_old_vs_new_lambdas.png
            - [outdir]/[outname]_splinefit.png
        * a new expanded-ensemble mdp file: [outname]_ee_optimized.mdp
              
    EXAMPLE
    Try this:
        $ cd ../examples
        $ python ../scripts/optimize_rest_coul_vdw_lambdas.py donepezil_rest_coul_vdW/ee.mdp donepezil_rest_coul_vdW/ee_test.dhdl.xvg opt donepezil_rest_coul_vdW
    """        

    # Parse input

    if len(sys.argv) < 5:
        print(usage)
        sys.exit(1)

    mdpfile      = sys.argv[1]
    dhdl_xvgfile = sys.argv[2]
    outname      = sys.argv[3]
    outdir       = sys.argv[4]

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    assert os.path.exists(outdir)

    # Optimize the lambdas
    new_rest_lambdas, new_coul_lambdas, new_vdw_lambdas = optimize_rest_coul_vdw_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, make_plots=True, save_plots=True, verbose=False)


    # print out new coul_lambdas, vdw_lambdas to std output, formatted like an mdp file
    print('### Optimized lambda values ###')
    print('')

    ## coul_lambdas_string
    coul_lambdas_string   = ' '.join(['%2.5f'%lam for lam in new_coul_lambdas])
    print(f'coul-lambdas         = {coul_lambdas_string}')

    ## vdw_lambdas_string
    vdw_lambdas_string    = ' '.join(['%2.5f'%lam for lam in new_vdw_lambdas])
    print(f'vdw-lambdas         = {vdw_lambdas_string}')

    ## rest_lambdas_string
    rest_lambdas_string   = ' '.join(['%2.5f'%lam for lam in new_rest_lambdas])
    print(f'restraint-lambdas         = {rest_lambdas_string}')

    print()

    new_rest_npyfile = os.path.join(outdir, f'{outname}_new_rest_lambdas.npy')
    np.save(new_rest_npyfile, new_rest_lambdas)
    print(f'Wrote: {new_rest_npyfile}')

    new_coul_npyfile = os.path.join(outdir, f'{outname}_new_coul_lambdas.npy')
    np.save(new_coul_npyfile, new_coul_lambdas)
    print(f'Wrote: {new_coul_npyfile}')

    new_vdw_npyfile = os.path.join(outdir, f'{outname}_new_vdw_lambdas.npy')
    np.save(new_vdw_npyfile, new_vdw_lambdas)
    print(f'Wrote: {new_vdw_npyfile}')

    # write [outname]_ee_optimized.mdp to file, to run for 10 ns
    from mdp_helpers import *
    e = ExpandedPrep(ligand_only=True, couple_moltype = 'LIG',
                     nsteps=5000000,  # 10000 ps = 10 ns simulation
                     nstexpanded=250,  # 500 fs = 0.5 ps per swap and dhdl snapshot (20000 total energy snaps in the dhdl file)
                     fep_lambdas=np.zeros(len(new_rest_lambdas)),   # This MUST be specified because nlambdas determined from it!
                     rest_lambdas=new_rest_lambdas,
                     coul_lambdas=new_coul_lambdas, 
                     vdw_lambdas=new_vdw_lambdas,
                     init_lambda_weights=np.zeros(len(new_rest_lambdas))  )
    ee_mdpfile = os.path.join(outdir, f'{outname}_ee_optimized.mdp')
    e.write_to_filename(ee_mdpfile)
    print(f'Wrote: {ee_mdpfile}')





