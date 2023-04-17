# README

The following script will take in an EE `*.mdp` and `*.dhdl_xvg` file  and write a new mdp file with optimizaed lambda

```
% python optimize_coul_vdw_lambdas.py --help

Usage:    python optimize_fep_lambdas.py mdpfile dhdl_xvgfile outname outdir

    DESCRIPTION
        This script will optimize the lambda values for all intermediates
        to minimize the total variance in P(\Delta u_ij) for neighboring thermodynamic ensembles

    OUTPUT
        * A mdpfile-compatible string with new coul-lambdas and vdw-lambdas printed to std output
        * Numpy arrays of the new values will be written to:
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
        $ python ../scripts/optimize_coul_vdw_lambdas.py Sulfamethazine_tau0_0_water/ee.mdp Sulfamethazine_tau0_0_water/ee.dhdl.xvg opt Sulfamethazine_tau0_0_water


