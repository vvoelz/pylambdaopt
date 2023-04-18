## ee_tools.py

## Functions and classes to set up and analyze expanded ensemble (EE) simulations

import os, sys
import numpy as np

import matplotlib.pyplot as plt
#  %matplotlib inline

import scipy 
from scipy import stats	

### Functions ###

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

def get_coul_vdw_lambdas(mdpfile):
    """Given an *.mdp file as input, extract the values of coul_lambdas and vdw_lambdas 

    RETURNS
    coul_lambdas      - numpy array of coul-lambdas
    vdw_lambdas

    """
    fin = open(mdpfile,'r')
    lines = fin.readlines()
    fin.close()

    coul_lambdas = None
    for line in lines:
        if (line.count('coul-lambdas') > 0) | (line.count('coul_lambdas') > 0):
          coul_string = line.split('=')[1].strip()
          #print ('coul_string=', coul_string)
          coul_lambdas = np.array([float(s) for s in coul_string.split()])
          #print ('coul_lambdas=', coul_lambdas)

    vdw_lambdas = None
    for line in lines:
        if (line.count('vdw-lambdas') > 0) | (line.count('vdw_lambdas') > 0): 
          vdw_string = line.split('=')[1].strip()
          #print ('vdw_string=', vdw_string)
          vdw_lambdas = np.array([float(s) for s in vdw_string.split()])
          #print ('vdw_lambdas=', vdw_lambdas)

    return coul_lambdas, vdw_lambdas


def get_dhdl_data(dhdl_xvgfile, verbose=True):
    r"""Read and parse the information in the dhdl file.
    
    RETURNS
    time_in_ps      - time in ps (1D np.array)
    thermo_states   - thermodynamic state indices (1D np.array)
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


    coul/vdw:

                                 @ s0 legend "Thermodynamic state"
                                 @ s1 legend "Total Energy (kJ/mol)"
                                 @ s2 legend "dH/d\xl\f{} fep-lambda = 0.0000"
                                 @ s3 legend "dH/d\xl\f{} coul-lambda = 0.0000"
                                 @ s4 legend "dH/d\xl\f{} vdw-lambda = 0.0000"
                                 @ s5 legend "dH/d\xl\f{} restraint-lambda = 1.0000"

    """
    
    import os
    assert os.path.exists(dhdl_xvgfile)
    
    # Read and parse the file
    fin = open(dhdl_xvgfile,"r")
    lines = fin.readlines()
    print ('tot lines=', len(lines))
    lines = lines[:2072]    #only using the first 2.ns data
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
    print(lines[0])
    print('ncols', ncols)
    
    time_in_ps, dhdl, thermo_states = [], [], []
    pV_values = []
    for line in lines:
        line_data_list = [float(s) for s in line.strip().split()] 
            
        # Skip line if it doesn't have the correct number of entries
        # (sometimes the I/O gets cutoff when writing the dhdl.xvg in it corrupts the data)
        if len(line_data_list) == ncols:
            time_in_ps.append(line_data_list[0])
            thermo_states.append(line_data_list[1])
            if pV_column_start == None:
                dhdl.append(line_data_list[dhdl_column_start:])
            else:
                # ignore the last column if it contains pV data 
                dhdl.append(line_data_list[dhdl_column_start:-1])
                pV_values.append(line_data_list[-1])
            
    time_in_ps = np.array(time_in_ps)
    dhdl = np.array(dhdl)
    thermo_states = np.array(thermo_states)
    
    print('time_in_ps.shape', time_in_ps.shape)
    print('dhdl.shape=', dhdl.shape)
    print('thermo_states=', thermo_states)

    if pV_column_start == None:
        return time_in_ps, thermo_states, dhdl, None
    else:
        return time_in_ps, thermo_states, dhdl, np.array(pV_values) 




def estimate_sigmas(dhdl, thermo_states, plot_data=True):
    """Using as input the Delta_U_ij energies from the dhdl array, 
    estimate the standard deviations P(U_{i-->i+1}) for neighboring ensembles.
    
    RETURNS
    sigmas   - a np.array() of standard deviations P(U_{i-->i+1}).
    """
    
    nlambdas = dhdl.shape[1]
    print('nlambdas', nlambdas)
    
    if plot_data:
        plt.figure(figsize=(6, 80))

    Delta_uij_values = []     
    sigmas = []
       
    for j in range(nlambdas-1):
    
        ## transitions from state 0 to 1 or 1 to 2, or 2 to 3 .... 

        Ind = (thermo_states == j)
        delta_u_ij = dhdl[Ind, j+1]       # forward delta_u only for neighbored ensembles

        Ind2 = (thermo_states == (j+1))
        delta_u_ji = dhdl[Ind2, j]       # forward delta_u only for neighbored ensembles

        #print ('lambda index=', j)
        #print ('delta_u_ij.shape=', delta_u_ij.shape)
      
        #Delta_uij_values.append(delta_u_ij)

        ### VAV debug
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




