import os, sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import scipy 
from scipy import stats	

from ee_tools import *

from optimizer import *
from plotting import *


def optimize_fep_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, optimize_K=False, make_plots=True, usePDF=True, verbose=True):
    """
    DESCRIPTION
        Optimize the lambda values for all intermediates by equally spacing them in thermodynamic length.

    INPUT
        mdpfile      - Input .mdp file
        dhdl_xvgfile - Input dhdl.xvg file
        outname      - basename for output file naming
        outdir       - Output directory

    PARAMETERS
        optimize_K   - If True, additionally optimize the number of intermediates K. Default: False
        make_plots   - If True, save plots to files.  Default: True
        usePDF       - If True, plots will be saved as PDF; otherwise use PNG. Default: True
        verbose      - If True, more verbose output will be printed 

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


    ##### optimize the numnber of alchemical intermediates K,  if specified 
    if optimize_K:

        #### optimize K  based on the mixing times

        optimal_K, K_values, P_acc_values, t2_values = o.optimize_K(alpha_values[0], alpha_values[-1], min_K=5, max_K=200)
        print('optimal_K =', optimal_K)

        # print('K_values', K_values)
        # print('P_acc_values', P_acc_values)
        # print('t2_values', t2_values)

        if make_plots:
            # make plots of the mixing time versus K 
            if usePDF:
                mixing_pdffile = os.path.join(outdir, f'{outname}_mixing.pdf')
                plot_mixing(K_values, P_acc_values, t2_values, mixing_pdffile)
            else:
                mixing_pngfile = os.path.join(outdir, f'{outname}_mixing.png')
                plot_mixing(K_values, P_acc_values, t2_values, mixing_pngfile)

        # Based on the optimal K, Optimize a new set of  \alpha_k values, k=1,...K
        adjusted_alpha_values = np.linspace(alpha_values[0], alpha_values[-1], optimal_K)
        new_alphas, traj_alphas = o.optimize_alphas(adjusted_alpha_values)

    else:
        # Optimize the \alpha_k values (without changing their number)
        new_alphas, traj_alphas = o.optimize_alphas(alpha_values)

   
    if make_plots:     
        # make plots of the optimization traces
        if usePDF:
            traces_pdffile = os.path.join(outdir, f'{outname}_opt_traces.pdf')
            plot_opt_traces(traj_alphas, o.L_spl, traces_pdffile)
        else:
            traces_pngfile = os.path.join(outdir, f'{outname}_opt_traces.png')
            plot_opt_traces(traj_alphas, o.L_spl, traces_pngfile)

    if make_plots:
        # make plots of old versus new alphas
        if usePDF:
            old_new_pdffile = os.path.join(outdir, f'{outname}_old_vs_new_alphas.pdf')
            plot_old_vs_new_alphas(alpha_values, new_alphas, o.L_spl, old_new_pdffile)
        else:
            old_new_pngfile = os.path.join(outdir, f'{outname}_old_vs_new_alphas.png')
            plot_old_vs_new_alphas(alpha_values, new_alphas, o.L_spl, old_new_pngfile)


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

    usage = """\
    This script will optimize the lambda values of all alchemical intermediates by equally
    spacing them in thermodynamic length.

    OUTPUT (saved in the specified outdir):
    * A mdpfile-compatible string with new fep-lambdas printed to std output
    * Numpy arrays of the new values will be written to:
        - [outdir]/[outname]_new_fep_lambdas.py
    * Graphs associated with the lambda optimization will be written as images:
        - [outdir]/[outname]_optimization_traces.pdf (or png)
        - [outdir]/[outname]_old_vs_new_lambdas.pdf (or png)
        - [outdir]/[outname]_splinefit.pdf (or png)

    Try this:
        $ cd ../examples
        $ python ../scripts/optimize_FEP_lambdas.py FEP_A8_A16F_unopt/EE.mdp FEP_A8_A16F_unopt/dhdl.xvg opt FEP_A8_A16F_unopt

    """

    # Initialize the parser using RawTextHelpFormatter to preserve newlines
    parser = argparse.ArgumentParser(description=usage, formatter_class=argparse.RawTextHelpFormatter)

    # Add positional arguments
    parser.add_argument("mdpfile", help="Input .mdp file")
    parser.add_argument("dhdl_xvgfile", help="Input dhdl.xvg file")
    parser.add_argument("outname", help="Output base name")
    parser.add_argument("outdir", help="Output directory")

    # Add an optional Boolean flag (verbose mode)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode for debugging")

    # Add an optional Boolean flag (verbose mode)
    parser.add_argument("--optimize_K", action="store_true", help="Additionally optimize the number of intermediates K")

    # If no arguments are provided, print usage and exit
    if len(sys.argv) == 1:  # No arguments provided
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()

    # Example usage of the verbose flag
    if args.verbose:
        print(f"mdp file: {args.mdpfile}")
        print(f"dhdl.xvg file: {args.dhdl_xvgfile}")
        print(f"Output Name: {args.outname}")
        print(f"Output Directory: {args.outdir}")


    # Access the arguments
    mdpfile = args.mdpfile
    dhdl_xvgfile = args.dhdl_xvgfile
    outname = args.outname
    outdir = args.outdir
    verbose = args.verbose
    optimize_K = args.optimize_K

    # create the outdir if necessary
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    assert os.path.exists(outdir)


    # Optimize the lambdas
    new_fep_lambdas = optimize_fep_lambdas(mdpfile, dhdl_xvgfile, outname, outdir, optimize_K=optimize_K, make_plots=True, usePDF=True, verbose=verbose)


    # print out new fep_lambdas, formatted like an mdp file
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


