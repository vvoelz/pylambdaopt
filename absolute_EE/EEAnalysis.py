import matplotlib.pyplot as plt
from matplotlib import cm, gridspec, rc
import numpy as np
import pandas as pd
import subprocess, collections
import glob, tqdm, os, re
import shutil

class EEAnalysis:
    def __init__(self, RL_directory= None, L_directory= None,
      last_nanoseconds= 0.5, mdp_file= 'prod.mdp', log_file= 'md.log',
      increment_threshold= 10):
        
        if RL_directory:
            self.path = RL_directory
            
        elif L_directory:
            self.path = L_directory
        else:
            print('EEAnalysis requires either RL_directory or L_directory.')
        self.RL_directory = RL_directory
        self.L_directory = L_directory
        self.last_nanoseconds = last_nanoseconds
        self.increment_threshold = increment_threshold
        self.mdp_file = mdp_file
        self.log_file = log_file
        
    def scrape_mdp(self, mdp_file):
        with open(mdp_file, 'r') as f:
            eq_dist, k = None, None
            for line in f:
                if 'nsteps' in line:
                    nsteps = int(line.split()[2])
                elif 'dt' in line:
                    timestep = int(float(line.split()[2])*1000)
                elif 'nstlog' in line:
                    log_freq = int(line.split()[2])
                elif 'coul-lambdas' in line:
                    lambdas = len(line.split()[2:])
                elif 'pull' in line and 'init' in line:
                    eq_dist = float(line.split()[2])
                elif 'pull' in line and 'k' in line.split()[0]:
                    k = float(line.split()[2])
                
        swaps_per_ns = int(1e6 / (timestep * log_freq))
        return swaps_per_ns, lambdas, eq_dist, k
                    
    def scrape_log(self, stride= 10):
        for project in [self.RL_directory, self.L_directory]:
            if not project:
                continue
            if project == self.RL_directory and project:
                self.path = self.RL_directory
                project_type = 'RL'
            elif project == self.L_directory and project:
                project_type = 'L'
                self.path = self.L_directory

            mdp_file = os.path.join(self.path,self.mdp_file)
            log_file = os.path.join(self.path,self.log_file)
            swaps_per_ns, lambdas, eq_dist, k = self.scrape_mdp(mdp_file)

            print(f'\nProcessing {self.path}')
            if not os.path.exists(log_file):
                print(f'Missing {log_file}!\n')
                continue
            increments, lambda_indices, frame_energies, free_energies, matrices = [],[],[],[],[]
            read_energies, read_matrix = False, False

            for line in open(log_file): # figure out the FE column
                if 'G(in kT)' in line:
                    fe_ndx = line.split().index('G(in')
                    break

            for line in tqdm.tqdm(open(log_file)):
                if 'Wang-Landau incrementor is:' in line:
                    increments.append(float(line.split()[-1]))
                elif '<<' in line:
                    lambda_indices.append(int(line.split()[0]))
                if f'1  0.000  0.000' in line and '0.00000' in line:
                    read_energies = True
                if read_energies:
                    frame_energies.append(float(line.split()[fe_ndx]))
                if len(frame_energies) == lambdas:
                    free_energies.append(frame_energies)
                    frame_energies, read_energies = [],False
                if '                     Transition Matrix' in line:
                    read_matrix = True
                    matrices.append([])
                if read_matrix:
                    matrices[-1].append(line.split()[:-1])
                    if len(matrices[-1]) == lambdas:
                        matrices[-1] = [[float(x)*100 for x in line] for line in matrices[-1][2:]]
                        read_matrix = False

            free_energies = np.asarray(free_energies)

#            last_energies = free_energies[len(free_energies)-int(
#              swaps_per_ns * self.last_nanoseconds):][:,-1]

#            end_variance = max(last_energies) - min(last_energies)

            print('Creating plot...\n')
            if 1: #end_variance < cutoff:

                np.save(f'{self.path}/feb.npy', free_energies[::stride])
                np.save(f'{self.path}/lam.npy', lambda_indices) #[::stride])
                np.save(f'{self.path}/inc.npy', increments) #[::stride])

            # plot lambda index over time colored by bias magnitude
                plt.figure(figsize=(18,10))
                gs = gridspec.GridSpec(2, 2, width_ratios=[5,2]) 
                lambda_indices = [int(x) for x in lambda_indices]
                occurrences = collections.Counter(lambda_indices)
                keys = list(occurrences.keys())
                occurrences = [occurrences[x] for x in occurrences]
                min_sampled_lam = keys[occurrences.index(min(occurrences))]
                weights = list(reversed(sorted(set(increments))))
                colors = cm.rainbow(np.linspace(1,0,len(weights)))
                scatter_colors = [colors[weights.index(x)] for x in increments]
                ax2 = plt.subplot(gs[0])
                ax2.scatter(range(len(free_energies)), free_energies[:,-1], s=6,
                 c=scatter_colors)
                ax2.set_ylim(free_energies[:,-1][-1]-20,free_energies[:,-1][-1]+20)
                ax3 = plt.subplot(gs[1])
                ax3.matshow(matrices[-1], cmap=plt.cm.viridis)
                ax0 = plt.subplot(gs[2])
                ax0.scatter(range(len(lambda_indices)), lambda_indices, s=6,
                 c=scatter_colors)
                ax2.axhline(y=np.average(free_energies[:,-1]))
                ax0.axhline(y=min_sampled_lam)
                ax0.set_xlabel('Frame Index')
                ax0.set_ylabel('Lambda Index')
                ax0.set_title(f'Lambda Trace for {self.path}')

                # plot increment trajectory
                ax1 = plt.subplot(gs[3])
                barWidth = 1
                increments = [np.log10(float(x)) for x in increments]
                ax1.scatter(range(len(increments)), increments, s=6, c=scatter_colors)
                ax1.set_xlabel('Frame Index')
                ax1.set_ylabel('log10 WL Increment')
                ax1.set_title('WL Increment Trajectory')
                plt.tight_layout()
                plt.savefig(f'{self.path}/EE.png')
                plt.close()
                
    def calculate_dG_rest(self, k, d=0.0, T=298.15, V0=1.66057789):
        """Return the analytical result for the free energy of *adding* a harmonic restraint
            $$ U_rest = (1/2)*k*(|x-x0| - d)^2

        INPUTS
            k - the harmonic force constant.  Units: (kJ mol^{-1} nm^{-2})

        PARAMETERS
            d  - the equilibrium restraint distance, Default: 0.0 (units nm)
            T  - the temperature, Default = 298.15 (units K)
            V0 - the standard volume in (nm^3)

        RETURNS
            dG - the free energy of *adding* the restraint in units kT

        ### Notes on standard volume
        >>> # Standard concentration is c0 = 1 M = 1 mol/L   (#particles / volume)
        >>> # Standard volume is V0 = 1/c0 = 1 L / 1 mol*N_A   (volume per particle)
        >>> N_A = 6.022e23  # Avogadro's number  (mol^{-1})
        >>> # 1 L = (1 dm)^3   = (10^8 nm)^3
        >>> V0 = (1.0e8)**3 / N_A
        >>> print('V0 =', V0, '(nm)^3')
        >>> #  V0 = 1.6605778811026237 (nm)^3
        """

        from scipy.special import erf

        N_A = 6.022e23    # mol{-1}
        kB_in_kJ_per_mol = 1.38064852e-23 * 1.e-3 * N_A # (J/K)*(kJ/J)*N_A - kJ/K
        kT = kB_in_kJ_per_mol * T  # in kJ/mol
        beta = 1.0/kT

        if d == 0.0:
            # calculation
            dG = -1.0 * ( (3./2.)*np.log( 2.0*np.pi/(beta*k) )  - np.log(V0) )
        else:
            dG = -1.0 * ( np.log( 4.*np.pi) + np.log( 1./4.*np.sqrt(8.0*np.pi/beta**3/k**3)*(1. + erf(d*(beta*k/2.0)**0.5))
                                                + 2.0*d/beta/k*np.exp(-1.*beta*k/2.0*d**2)
                                                + (d**2)/2.*np.sqrt(2.0*np.pi/beta/k)*(1. + erf(d*(beta*k/2.0)**0.5))  )
                                        - np.log(V0) )

        return dG


    def analyze(self, verbose= True):
        
        
        scraped_RL_feb_path = f'{self.RL_directory}/feb.npy'
        scraped_L_feb_path = f'{self.L_directory}/feb.npy'
        if verbose:
            print('\nLooking for scraped_RL_feb_path', scraped_RL_feb_path, os.path.exists(scraped_RL_feb_path))
            print('Looking for scraped_L_feb_path', scraped_L_feb_path, os.path.exists(scraped_L_feb_path))

        if not os.path.exists(scraped_RL_feb_path):
            print(f"Can't find {scraped_RL_feb_path}.")
            return
        if not os.path.exists(scraped_L_feb_path):
            print(f"Can't find {scraped_L_feb_path}.")
            return
        
    # Note: all dG are in units kT 
    
    ####################### 
    # Compute dG1 -  the free energy of restraining the ligand in a solution
    #                at standard concentration (analytical)
        mdp_file = os.path.join(self.RL_directory, self.mdp_file)
        swaps_per_ns, lambdas, eq_dist, k = self.scrape_mdp(mdp_file)

        dG1 = self.calculate_dG_rest(k, d=eq_dist, T=298.15, V0=1.66057789)
    
    ####################### 
    # Compute dG2 -   free energy of decoupling the ligand in solution (EE)
    
        L_increments = np.load(f'{self.L_directory}/inc.npy')
        L_fe = np.load(f'{self.L_directory}/feb.npy')[:,-1]

        for x,i in enumerate(L_increments):
            if i < self.increment_threshold:
                break

        dG2, dG2_sigma = np.average(L_fe[x:]), np.std(L_fe[x:])

    ####################### 
    # Compute dG3 -   free energy of *coupling* the ligand bound to the receptor
    #                 in the presence of restraints (EE)
    # dG4 is included in this because we scale off restraint during EE!
    
        RL_increments = np.load(f'{self.RL_directory}/inc.npy')
        RL_fe = np.load(f'{self.RL_directory}/feb.npy')[:,-1]

        for x,i in enumerate(RL_increments):
            if i < self.increment_threshold:
                last_frames = x*10 # un-stride for later
                break

        dG3, dG3_sigma = -np.average(RL_fe[x:]), np.std(RL_fe[x:])
        
        dG_binding = dG1 + dG2 + dG3 #+ dG4_BAR
        dG_binding_sigma = np.sqrt( dG2_sigma**2 + dG3_sigma**2 )
        Kd = np.exp(dG_binding)

        log_Kd_sigma = dG_binding_sigma/np.log(10)

        print('------------------------------------------------------------')
        print(f'RL: {self.RL_directory}')
        print(f'L:  {self.L_directory}')
        print('------------------------------------------------------------')
        print(f'dG1 = \t{dG1}')
        print(f'dG2 = \t{dG2} +/- {dG2_sigma}')
        print(f'dG3 = \t{dG3} +/- {dG3_sigma}')
        print('-----------')
        print(f'dG_binding (units RT)   = {dG_binding}')
        print(f'dG_binding_uncertainty  = {dG_binding_sigma}')
        print(f'Kd                      = %3.2e M'%Kd)
        print(f'log_10 Kd               = %3.2f'%np.log10(Kd) )
        print(f'log_10 Kd uncertainty   = %3.2f'%log_Kd_sigma )
        print('------------------------------------------------------------')
        print('\n')
        
        results = [dG1, dG2, dG3, dG_binding, np.log10(Kd),
             dG2_sigma, dG3_sigma, dG_binding_sigma, log_Kd_sigma,
             RL_increments[-1], L_increments[-1], np.shape(RL_increments)[0]/100, np.shape(L_increments)[0]/10]
        
        return results
