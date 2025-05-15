## hrex_tools.py

## Functions and classes to set up and analyze hamiltonian replica exchange (HREX) simulations
## The Functions are truncated from ee_tools, only include the optimized for lambda, not include the K-intermediated

import os, sys, shutil
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	

### Functions ###

def copy_dhdl_files(src_base):
    # Ensure destination base directory exists
    
    dst_base = os.path.join(src_base, 'dhdl_files')
    os.makedirs(dst_base, exist_ok=True)

    # Find all dhdl* files under state_* directories
    dhdl_files = glob(os.path.join(src_base, 'state_*', 'dhdl*'))

    for file_path in dhdl_files:
        # Extract the subdirectory (e.g., state_0)
        state_dir = os.path.basename(os.path.dirname(file_path))

        # Create the corresponding directory inside the destination
        dst_dir = os.path.join(dst_base, state_dir)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy the file
        dst_path = os.path.join(dst_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dst_path)
        print(f"Copied {file_path} → {dst_path}")
    return dst_base


def get_fep_lambdas(mdpfile):
    """Given an *.mdp file as input, extract the values of fep-lambdas
    RETURNS
    fep_lambdas      - numpy array of fep-lambdas
    NOTE: for the alchemical transformations, the lambda=0 state is ligand L, 
          and lambda=1 is ligand L*.
    """
    fin = open(mdpfile,'r')
    lines = fin.readlines()
    fin.close()
   
    fep_lambdas = None
    for line in lines:
        fields = line.strip().split()
        if len(fields) > 0:
            if (fields[0].count('fep-lambdas') > 0) | (fields[0].count('fep_lambdas') > 0):
                fep_string = line.split('=')[1].strip() 
                #print ('fep_string=', fep_string)
                fep_lambdas = np.array([float(s) for s in fep_string.split()])
                #print ('fep_lambdas=', fep_lambdas)

    return fep_lambdas

def get_dhdl_data_hrex(dhdl_xvgfile, verbose=True):
    r"""Read and parse the information in the dhdl file path.
    The path should be have format like:

    dhdl_files_path/ state_*/dhdl_xvgfile
    the * are range from fep_lambdas, starting from 0 
    
    RETURNS
    time_in_ps      - time in ps (1D np.array)
    dhdl            - delta_Uij (np.array of shape (N,K))  
                      where N is snapshots and K is number of thermodynamic states
    pV              - a 1D np.array of the pV values, if simulation was NPT;
                      None, otherwise
                     
    
    **NOTE 1**:
    For FEP vs couul/vdw the number of columns in the dhdl.xvg file will be different!
    Here we look for the first appearance of "\xD\f{}H" ...

    time         ---> Column 0 is time in ps.                         
    thermo_index ---> Column 1 = @ s0 legend "Thermodynamic state"
                      Column 2 = @ s1 legend "Total Energy (kJ/mol)"
                      Column 3 = @ s2 legend "dH/d\xl\f{} fep-lambda = 0.0000"
    
    dU_ij starts ---> Column 4 = @ s3 legend "\xD\f{}H \xl\f{} to 0.0000"
                      Column 5 = @ s4 legend "\xD\f{}H \xl\f{} to 0.0020"
                      Column 6 = @ s5 legend "\xD\f{}H \xl\f{} to 0.0040"

                         ...  
                                 @ s104 legend "\xD\f{}H \xl\f{} to (0.0000, 1.0000, 1.0000, 1.0000)"
                         --->    @ s105 legend "pV (kJ/mol)"

    **NOTE 2**: if NPT, there will be an extra line for pV!

    **NOTE 3**: VAV: Apr 20, 2024
    If the file is rest->coul->vdW, then there will be an additional rest line

    coul/vdw:

                                 @ s0 legend "Thermodynamic state"
                                 @ s1 legend "Total Energy (kJ/mol)"
                                 @ s2 legend "dH/d\xl\f{} fep-lambda = 0.0000"
                                 @ s3 legend "dH/d\xl\f{} coul-lambda = 0.0000"
                                 @ s4 legend "dH/d\xl\f{} vdw-lambda = 0.0000"
                                 @ s5 legend "dH/d\xl\f{} restraint-lambda = 1.0000"

    **NOTE 4**: Starwing May 15 2025
    
    This function is modified for HREX only, where the lamba-optimized are only tested on fep-lambda. 
    The difference between HREX and EE is that there is no thermo-index in HREX, however, all the dhdl value are writted parallel across the state. 
    Thus, when extracted information, we have to extracte them all. 
    And this function is only used for extracted one dhdl_xvg_file 
    The **NOTE 3** May not applied to HREX, however, this need to be tested and updated. 

    time         ---> Column 0 is time in ps.                         
    thermo_index ---> Column 1 = @ s1 legend "Total Energy (kJ/mol)"
                      Column 2 = @ s2 legend "dH/d\xl\f{} fep-lambda = 0.0000"
    
    dU_ij starts ---> Column 3 = @ s3 legend "\xD\f{}H \xl\f{} to 0.0000"
                      Column 4 = @ s4 legend "\xD\f{}H \xl\f{} to 0.0020"
                      Column 5 = @ s5 legend "\xD\f{}H \xl\f{} to 0.0040"

    """
    # Read and parse the file
    fin = open(dhdl_xvgfile,"r")
    lines = fin.readlines()
    if verbose: 
        print ('tot lines=', len(lines))
    lines = lines[:2072]    #only using the first 2.ns data
    if verbose:
        print ('tot lines=', len(lines))
    fin.close()

    # Search headers to find which column starts the dhdl data
    dhdl_column_start = None
    for line in lines:
        # Looking for line like this: "@ s5 legend "\xD\f{}H \xl\f{} to (0.0000, 0.0000, 0.0000)"
        if line.count('"\\xD\\f{}H') > 0:
            dhdl_column_start = int(line.split(' ')[1].replace('s','')) + 1
            break

    if verbose:
        print('dhdl data starts at column:', dhdl_column_start)


    # Search headers to find pV, if it exists"
    pV_column_start = None
    for line in lines:
        # Looking for line like this: 's105 legend "pV (kJ/mol)"'
        if line.count('pV (kJ/mol') > 0:
            pV_column_start = int(line.split(' ')[1].replace('s','')) + 1
            break
    
    if verbose: 
        if pV_column_start != None:
            print('Found pV data at column:', pV_column_start)
        else:
            print('No pV data found.')


    # Get rid of all the header lines
    i = 0
    while i < len(lines):
        if (lines[i][0] == '#') or (lines[i][0] == '@'):
            lines.pop(i)
        else:
            i+=1

    # find the correct number of entries from the first line
    ncols = len(lines[0].strip().split())
    if verbose:
        print(lines[0])
        print('ncols', ncols)



    time_in_ps, dhdl = [], []
    pV_values = []
    for line in lines:
        line_data_list = [float(s) for s in line.strip().split()] 
            
        # Skip line if it doesn't have the correct number of entries
        # (sometimes the I/O gets cutoff when writing the dhdl.xvg in it corrupts the data)
        if len(line_data_list) == ncols:
            time_in_ps.append(line_data_list[0])
            if pV_column_start == None:
                dhdl.append(line_data_list[dhdl_column_start:])
            else:
                # ignore the last column if it contains pV data 
                dhdl.append(line_data_list[dhdl_column_start:-1])
                pV_values.append(line_data_list[-1])
            
    time_in_ps = np.array(time_in_ps)
    dhdl = np.array(dhdl)


    
    if pV_column_start == None:
        return time_in_ps, dhdl, None
    else:
        return time_in_ps, dhdl, np.array(pV_values) 
    



def get_all_state_dhdl(dhdl_xvgfile_path, lambdas,  verbose=True):

    num_lambda = len(lambdas) # Collect number of thermodynamics lambda state 
    dhdl_all_state = []

    for state in range(num_lambda):
        # Parce the file in that lambda state 
        state_path = f"{dhdl_xvgfile_path}/state_{state}/"
        dhdl_xvgfile = ' '.join([
        f"{state_path}/{file}" for file in os.listdir(state_path)
        if 'dhdl' in file and 'xvg' in file
    ])
        if verbose:
            print('dhdl_xvgfile', dhdl_xvgfile)

        time_in_ps, dhdl, pV_values = get_dhdl_data_hrex(dhdl_xvgfile, verbose=verbose)

        dhdl_all_state.append(dhdl)

    dhdl_all_state = np.array(dhdl_all_state)
    
    print('time_in_ps.shape', time_in_ps.shape)
    print('dhdl_all_state.shape.shape', dhdl_all_state.shape)
    
    return time_in_ps, dhdl_all_state, pV_values


def estimate_sigmas(dhdl_all_state, plot_data=True):
    """Using as input the Delta_U_ij energies from the dhdl_all_state array, 
    estimate the standard deviations P(U_{i-->i+1}) for neighboring ensembles.
    
    RETURNS
    sigmas   - a np.array() of standard deviations P(U_{i-->i+1}).
    """


    nlambdas = dhdl_all_state.shape[2]
    print('nlambdas', nlambdas)

    Delta_uij_values = []     
    sigmas = []

    for j in range(nlambdas-1):
        dhdl_state_i = dhdl_all_state[j]
        dhdl_state_j = dhdl_all_state[j+1]
        delta_u_ij = dhdl_state_i[:, j+1]
        delta_u_ji = dhdl_state_j[:, j]

        print ('lambda index=', j)
        print ('delta_u_ij.shape=', delta_u_ij.shape)

        print('Are any delta_u_ij values nan?')
        print(delta_u_ij)
        print('Are any delta_u_ji values nan?')
        print(delta_u_ji)

        mu_ij, sigma_ij = scipy.stats.norm.fit(delta_u_ij)
        mu_ji, sigma_ji = scipy.stats.norm.fit(delta_u_ji) 

        sigma = ( sigma_ij + sigma_ji ) / 2.0
        sigmas.append(sigma)

        delta_u_bins = np.arange(-15., 15., 0.2)
        counts, bin_edges = np.histogram(delta_u_ij, bins=delta_u_bins)
        counts = counts/counts.sum() # normalize
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2.0

        if plot_data:
            plt.subplot(nlambdas-1, 1, j+1)
            plt.step(bin_centers, counts, label='$\Delta u_{%d \\rightarrow %d} \sigma$=%.2f'%(j,j+1,sigma))
            #plt.xlabel('$\Delta u_{%d \\rightarrow %d}$'%(j, j+1))
            plt.legend(loc='best')
        
    if plot_data:
        plt.tight_layout()
        plt.show()

    ## VAV: hot fix for non-sampled lambda (sigma = nan), or sampled only once (sigma = 0)
    max_sigma = max(sigmas)
    for i in range(len(sigmas)):
        if (sigmas[i] == 0) or np.isnan(sigmas[i]):
            sigmas[i] = max_sigma

    return np.array(sigmas)
