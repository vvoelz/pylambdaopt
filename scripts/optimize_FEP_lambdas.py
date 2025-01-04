import os, sys
import numpy as np
import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	

from ee_tools import *

from optimizer import *
from plotting import *


def optimize_fep_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, make_plots=True, save_plots=True, verbose=True):
    """
    DESCRIPTION
        Optimize the lambda values for all intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    RETURNS
        fep-lambdas (their optimized values)

    """


    # For a simple FEP alchemical transformation, there is only one lambda parameter,
    # so that phi(\alpha_k) = \lambda_k

    lambdas = get_fep_lambdas(mdpfile)

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
    phi = o.create_phi(labels=['FEP'], ascending=[True])

    print('new_alphas', new_alphas)

    # perform the mapping and return the new lambdas
    new_lambdas = phi(new_alphas)

    return new_lambdas


def write_optimized_fep_mdp(new_fep_lambdas, input_mdpfile, output_mdpfile):
    """Replace the "fep-lambdas" line in the mdpfile with the optimized values."""

    # Read in lines from the input mdpfile
    fin = open(input_mdpfile, 'r')
    lines = fin.readlines()
    fin.close()

    ## Format the new fep_lambdas_string for the mdp file
    fep_lambdas_string   = ' '.join(['%2.5f'%lam for lam in new_fep_lambdas])
    fep_lambdas_line = f'fep-lambdas         = {fep_lambdas_string}\n'

    # Find the 'fep-lambdas' line 
    for i in range(len(lines)):
        line = lines[i]
        fields = line.strip().split()
        if len(fields) > 0:
            if (fields[0].count('fep-lambdas') > 0) | (fields[0].count('fep_lambdas') > 0):
                lines[i] = fep_lambdas_line

    # Write the output mdpfile
    fout = open(output_mdpfile, 'w')
    fout.writelines(lines)
    fout.close()




########################

if __name__ == '__main__':

    usage  = """Usage:    python optimize_fep_lambdas.py mdpfile dhdl_xvgfile outname outdir

    DESCRIPTION
        This script will optimize the lambda values for all intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    OUTPUT
        * A mdpfile-compatible string with new fep-lambdas printed to std output
        * Numpy arrays of the new values will be written to:
            - [outdir]/[outname]_new_fep_lambdas.py
        * graphs associated with the lambda optimization will be written as images:
            - [outdir]/[outname]_optimization_traces.png
            - [outdir]/[outname]_old_vs_new_lambdas.png
            - [outdir]/[outname]_splinefit.png
        * a new expanded-ensemble mdp file: [outname]_ee_optimized.mdp
              
    EXAMPLE
    Try this:
        $ cd ../examples
        $ python ../scripts/optimize_fep_lambdas.py relative_FEP_R3E_miniprotein/EE.mdp relative_FEP_R3E_miniprotein/dhdl.xvg opt relative_FEP_R3E_miniprotein
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
    new_fep_lambdas = optimize_fep_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, make_plots=True, save_plots=True, verbose=False)


    # print out new fep_lambdass to std output, formatted like an mdp file
    print('### Optimized lambda values ###')
    print('')
    ## fep_lambdas_string
    fep_lambdas_string   = ' '.join(['%2.5f'%lam for lam in new_fep_lambdas])
    print(f'fep-lambdas         = {fep_lambdas_string}')
    print()


    new_fep_npyfile = os.path.join(outdir, f'{outname}_new_fep_lambdas.npy')
    np.save(new_fep_npyfile, new_fep_lambdas)
    print(f'Wrote: {new_fep_npyfile}')


    # write [outname]_ee_optimized.mdp to file
    ee_mdpfile = os.path.join(outdir, f'{outname}_ee_optimized.mdp')
    write_optimized_fep_mdp(new_fep_lambdas, mdpfile, ee_mdpfile)
    print(f'Wrote: {ee_mdpfile}')


