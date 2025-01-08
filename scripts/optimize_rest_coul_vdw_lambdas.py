import os, sys
import numpy as np
import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	

from ee_tools import *
from optimizer import *
from plotting import *


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
        spline_pngfile = os.path.join(outdir, f'{outname}_splinefit.png') 
        spline_pdffile = os.path.join(outdir, f'{outname}_splinefit.pdf')
        plot_spline(alpha_values, L_values, o.L_spl, o.L_spl_1d, spline_pdffile)


    # Optimize the \alpha_k values
    new_alphas, traj_alphas = o.optimize_alphas(alpha_values)

   
    if make_plots:     
        # make plots of the optimization traces
        traces_pngfile = os.path.join(outdir, f'{outname}_opt_traces.png')
        traces_pdffile = os.path.join(outdir, f'{outname}_opt_traces.pdf')
        plot_opt_traces(traj_alphas, o.L_spl, traces_pdffile)

    
    if make_plots:
        # make plots of old versus new alphas
        old_new_pngfile = os.path.join(outdir, f'{outname}_old_vs_new_alphas.png')
        old_new_pdffile = os.path.join(outdir, f'{outname}_old_vs_new_alphas.pdf')
        plot_old_vs_new_alphas(alpha_values, new_alphas, o.L_spl, old_new_pdffile)


    # create a path function $\phi(\alpha) \rightarrow \vec{\lambda} to map the alphas back to lambdas
    phi = o.create_phi(labels=['rest', 'coul', 'vdw'], ascending=[True, True, True])

    print('new_alphas', new_alphas)

    # perform the mapping and return the new lambdas
    new_lambdas = phi(new_alphas)
    new_rest_lambdas, new_coul_lambdas, new_vdw_lambdas = new_lambdas[:,0], new_lambdas[:,1], new_lambdas[:,2]

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





