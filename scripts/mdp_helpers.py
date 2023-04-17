import os, sys
import time
import numpy as np


### Functions

def write_gromacs_min_mdp(filename, steps=100000):
    """Writes a mdp file for minimization in gromacs."""

    
    with open(filename, 'w') as f2:
        f2.write(f'''integrator          = steep
nsteps                 = {steps}
emtol                  = 10
emstep                 = 0.01
; Bond parameters
continuation           = no                          ; Initial simulation
; constraint_algorithm   = lincs                       ; holonomic constraints
constraints            = none
;constraints            = h-bonds                     ; all bonds (even heavy atom-H bonds) constrained
lincs_iter             = 1                           ; accuracy of LINCS
lincs_order            = 4                           ; also related to accuracy
; Neighborsearching
ns_type                = grid                        ; search neighboring grid cels
nstlist                = 10                          ; 20 fs
rlist                  = 1.4                         ; short-range neighborlist cutoff (in nm)
rcoulomb               = 1.4                         ; short-range electrostatic cutoff (in nm)
rvdw                   = 1.4                         ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype            = PME                         ; Particle Mesh Ewald for long-range electrostatics
pme_order              = 4                           ; cubic interpolation
fourierspacing         = 0.16    ''')
    f2.close()

def write_gromacs_preequil_mdp(filename):
    ''' writes a gromacs mdp file for PRE-equilibration in the NVT ensemble using md-vv'''
    with open(filename, 'w') as f:
        f.write(f'''; Run parameters
integrator             = md-vv                       ; leap-frog integrator
nsteps                 = 50000                       ; 2 * 500000 = 100 ps
dt                     = 0.001                       ; 1 fs -- hot fix VAV
; Output control
nstxout                = 1000                        ; save coordinates every 2 ps
nstvout                = 1000                        ; save velocities every 2 ps
nstenergy              = 1000                        ; save energies every 2 ps
nstlog                 = 1000                        ; update log file every 2 ps
; Bond parameters
continuation           = no                          ; Initial simulation
; constraint_algorithm   = lincs                       ; holonomic constraints
; constraints            = none
constraints            = h-bonds                     ; h-bonds 
lincs_iter             = 1                           ; accuracy of LINCS
lincs_order            = 4                           ; also related to accuracy
; Neighborsearching
ns_type                = grid                        ; search neighboring grid cels
nstlist                = 10                          ; 20 fs
rlist                  = 1.4                         ; short-range neighborlist cutoff (in nm)
rcoulomb               = 1.4                         ; short-range electrostatic cutoff (in nm)
rvdw                   = 1.4                         ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype            = PME                         ; Particle Mesh Ewald for long-range electrostatics
pme_order              = 4                           ; cubic interpolation
fourierspacing         = 0.16                        ; grid spacing for FFT
; Temperature coupling is on
tcoupl                 = V-rescale                   ; Weak coupling for initial equilibration
tc-grps                = System                      ; one group is more group
tau_t                  = 0.1                         ; time constant, in ps
ref_t                  = 298.15                      ; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl                 = no                          ; NVT
ref_p                  = 1.0                         ; reference pressure (in bar)
compressibility        = 4.5e-5                      ; isothermal compressibility, bar^-1
refcoord_scaling       = com
; Periodic boundary conditions
pbc                    = xyz                         ; 3-D PBC
; Dispersion correction
DispCorr               = EnerPres                    ; account for cut-off vdW scheme
; Velocity generationd
gen_vel                = yes                         ; Velocity generation is on
gen_temp               = 298.15                      ; temperature for velocity generation
gen_seed               = -1                          ; random seed
; These options remove COM motion of the system
nstcomm                = 10
comm-mode              = Linear
comm-grps              = System''')
    f.close()


def write_gromacs_equil_mdp(filename):
    ''' writes a gromacs mdp file for equilibration'''
    with open(filename, 'w') as f:
        f.write(f'''; Run parameters
integrator             = md-vv                        ; leap-frog integrator
nsteps                 = 100000                      ; 2 * 100000 = 200 ps
dt                     = 0.002                       ; 2 fs
; Output control
nstxout                = 1000                        ; save coordinates every 2 ps
nstvout                = 1000                        ; save velocities every 2 ps
nstenergy              = 1000                        ; save energies every 2 ps
nstlog                 = 1000                        ; update log file every 2 ps
; Bond parameters
continuation           = no                          ; Initial simulation
; constraint_algorithm   = lincs                       ; holonomic constraints
; constraints            = none
constraints            = h-bonds                     ; h-bonds 
lincs_iter             = 1                           ; accuracy of LINCS
lincs_order            = 4                           ; also related to accuracy
; Neighborsearching
ns_type                = grid                        ; search neighboring grid cels
nstlist                = 10                          ; 20 fs
rlist                  = 1.4                         ; short-range neighborlist cutoff (in nm)
rcoulomb               = 1.4                         ; short-range electrostatic cutoff (in nm)
rvdw                   = 1.4                         ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype            = PME                         ; Particle Mesh Ewald for long-range electrostatics
pme_order              = 4                           ; cubic interpolation
fourierspacing         = 0.16                        ; grid spacing for FFT
; Temperature coupling is on
tcoupl                 = V-rescale                   ; Weak coupling for initial equilibration
tc-grps                = System                      ; one group is more group
tau_t                  = 0.1                         ; time constant, in ps
ref_t                  = 298.15                      ; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl                 = Berendsen                   ; Pressure coupling on in NPT, also weak coupling
pcoupltype             = isotropic                   ; uniform scaling of x-y-z box vectors
tau_p                  = 2.0                         ; time constant, in ps
ref_p                  = 1.0                         ; reference pressure (in bar)
compressibility        = 4.5e-5                      ; isothermal compressibility, bar^-1
refcoord_scaling       = com
; Periodic boundary conditions
pbc                    = xyz                         ; 3-D PBC
; Dispersion correction
DispCorr               = EnerPres                    ; account for cut-off vdW scheme
; Velocity generationd
gen_vel                = yes                         ; Velocity generation is on
gen_temp               = 298.15                      ; temperature for velocity generation
gen_seed               = -1                          ; random seed
; These options remove COM motion of the system
nstcomm                = 10
comm-mode              = Linear
comm-grps              = System''')
    f.close()



### Classes

class ExpandedPrep(object):
    """A class stucture for creating and writing standardized *.mdp files to do
    expanded-ensemble GROMACS 2018+ simualtions ."""

    def __init__(self,  ligand_only=False,
                        couple_moltype = 'LIG',
			pull_group1_name = 'a1-Receptor', 
                        pull_group2_name = 'a2-Ligand', 
                        pull_geometry    = 'distance',
                        pull_coord1_k    = 800.0, 
                        pull_coord1_init  = 0.0,   # an initial guess; gets measured in make_fep_ready.py
                        fep_lambdas      = np.array( 99*[0.0] ),

                        coul_lambdas     = np.array( [ 0.0000, 0.0345, 0.0706, 0.1040, 0.1435, 0.1873, 0.2263, 0.2661, 0.3081,
                                                       0.3468, 0.3942, 0.4405, 0.4846, 0.5361, 0.5914, 0.6478, 0.6984, 0.7568,
                                                       0.8178, 0.8702, 0.9259, 0.9828, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 ]),
                        vdw_lambdas      = np.array( [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                       0.0000, 0.0000, 0.0000, 0.0000, 0.0226, 0.0570, 0.0762, 0.0949, 0.1190,
                                                       0.1460, 0.1655, 0.1901, 0.2126, 0.2338, 0.2533, 0.2711, 0.2882, 0.3022,
                                                       0.3175, 0.3335, 0.3485, 0.3644, 0.3852, 0.4033, 0.4169, 0.4308, 0.4486,
                                                       0.4643, 0.4808, 0.5004, 0.5138, 0.5235, 0.5338, 0.5469, 0.5596, 0.5718,
                                                       0.5851, 0.5975, 0.6095, 0.6238, 0.6357, 0.6447, 0.6533, 0.6626, 0.6725,
                                                       0.6827, 0.6916, 0.6990, 0.7058, 0.7127, 0.7204, 0.7290, 0.7376, 0.7458,
                                                       0.7536, 0.7611, 0.7685, 0.7761, 0.7839, 0.7917, 0.7991, 0.8062, 0.8133,
                                                       0.8207, 0.8283, 0.8359, 0.8432, 0.8502, 0.8572, 0.8642, 0.8718, 0.8807,
                                                       0.8920, 0.9047, 0.9179, 0.9320, 0.9435, 0.9536, 0.9652, 0.9812, 1.0000 ]),
                        rest_lambdas     = np.array( [ 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                                       1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 ]),
                        init_lambda_weights = np.zeros(99),
                        init_lambda_state   = 0,
                        wl_increment_in_kT  = 10.0,
                        dt                  = 0.002, # ps
                        nsteps              = 5000000,
                        nstexpanded         = 250,      # Default is 0.5 ps attempted swaps, also must match nstdhdl !! 
                        temperature         = 298.15 ):



        """Initialize the class with default options.

        OPTIONS

        ligand_only -   If True, then the mdp will be written WITHOUT the pull code and without
                        reference to any protein.

                        The work unit will also be 10x longer

        [ to do -- add other options here ] 

        NOTES

        The *.mdp files that this class reads and write MUST have standard naming conventions

        For the ligand,
            * the Gromacs topology `*.top` MUST have a moleculetype `LIG`
            * the index file `*.ndx` MUST have an index group `LIG` 

        ### Protein-ligand simulations ###

        For a protein-ligand simulation, there is a tether between the ligand to the protein,
        and we need to have prepared an index file with the atom groups `a1-Protein`, `a2-Ligand` and
        `Restraint-Distance`, define as in the following example:
            ....
            [ a1-Protein ]
            678
            [ a2-Ligand ]
            1564
            [ Restraint-Distance ]
            678 1564

        Moreover, there needs to be atom groups `Protein` and `non-Protein`, which the thermostat will control separately.

        ### Ligand-only simulations ###

        There needs to be atom groups `Water` and `non-Water`, which the thermostat will control separately.


        OPTIONS

        ligand_only -   If True, then the mdp will be written WITHOUT the pull code and without
                        reference to any protein."""

        self.ligand_only      = ligand_only 
        self.couple_moltype   = couple_moltype 
        self.pull_group1_name = pull_group1_name
        self.pull_group2_name = pull_group2_name
        self.pull_geometry    = pull_geometry
        self.pull_coord1_k    = pull_coord1_k     # the spring constant in kJ/nm^2
        self.pull_coord1_init = pull_coord1_init  # the equilibrium distance between the pull group 1 and 2
        self.fep_lambdas      = fep_lambdas
        self.rest_lambdas     = rest_lambdas
        self.nlambdas         = len(self.fep_lambdas)
        self.fep_lambdas_string   = ' '.join(['%2.5f'%fep_lambdas[i] for i in range(self.nlambdas)])
        self.rest_lambdas_string   = ' '.join(['%2.5f'%rest_lambdas[i] for i in range(self.nlambdas)])

        self.init_lambda_weights  = init_lambda_weights
        self.init_lambda_weights_string =  ' '.join(['%2.5f'%init_lambda_weights[i] for i in range(self.nlambdas)])

        self.coul_lambdas     = coul_lambdas
        self.coul_lambdas_string   = ' '.join(['%2.5f'%coul_lambdas[i] for i in range(self.nlambdas)])

        self.vdw_lambdas     = vdw_lambdas
        self.vdw_lambdas_string   = ' '.join(['%2.5f'%vdw_lambdas[i] for i in range(self.nlambdas)])

        self.init_lambda_state    = init_lambda_state
        self.wl_increment_in_kT   = wl_increment_in_kT

        self.dt               = dt
        self.nsteps           = nsteps
        self.nstexpanded      = nstexpanded
        self.temperature      = temperature

        # Here's a random number seed, in case we need it:
        self.randseed         = np.random.randint(1e5)


    def report(self):
        """Print a report of the settings."""

        print('self.ligand_only', self.ligand_only)
        print('self.couple_moltype', self.couple_moltype)
        print('self.pull_group1_name', self.pull_group1_name)
        print('self.pull_group2_name', self.pull_group2_name)
        print('self.pull_coord1_k', self.pull_coord1_k)
        print('self.pull_coord1_init', self.pull_coord1_init)
        print('self.fep_lambdas', self.fep_lambdas)
        print('self.rest_lambdas', self.rest_lambdas)
        print('self.nlambdas', self.nlambdas)
        print('self.fep_lambdas_string', self.fep_lambdas_string)
        print('self.rest_lambdas_string', self.rest_lambdas_string)
        print('self.init_lambda_weights', self.init_lambda_weights)
        print('self.init_lambda_weights_string', self.init_lambda_weights_string)
        print('self.init_lambda_state', self.init_lambda_state)
        print('self.wl_increment_in_kT', self.wl_increment_in_kT)
        print('self.dt', self.dt) 
        print('self.nsteps', self.nsteps)
        print('self.temperature', self.temperature)
        print('self.randseed', self.randseed)
         

    def generate_mdp_text(self):
        """Builds and returns a string corresponding to the mdpfile contents."""

        self.header_desc = ";       generated from expanded_ensemble_mdpfile() on %s \n;\n;\n"%time.asctime()

        self.mdp_text = ''

        if self.ligand_only:
            self.mdp_text +=  """; Run control
integrator               = md-vv
tinit                    = 0
dt                       = {dt:.4f}
nsteps                   = {nsteps}
comm-mode                = Linear
nstcomm                  = 1

; Output control
nstlog                   = 2500
nstcalcenergy            = 1
nstenergy                = 250000
nstxout-compressed       = 250000        ; save xtc coordinates
nstxout                  = 2500000       ; save all coordinates
nstvout                  = 2500000       ; save all velocities
compressed-x-precision   = 1000

; This selects the subset of atoms for the .xtc file. You can
; select multiple groups. By default all atoms will be written.
compressed-x-grps        = LIG

; Selection of energy groups
energygrps               = System
"""
        else:
            self.mdp_text +=  """; Run control
integrator               = md-vv
tinit                    = 0
dt                       = {dt:.4f}
nsteps                   = {nsteps}
comm-mode                = Linear
nstcomm                  = 1

; Output control
nstlog                   = 2500
nstcalcenergy            = 1
nstenergy                = 25000        ; save edr every 100 ps
nstxout-compressed       = 25000        ; save xtc coordinates every 100 ps
nstxout                  = 250000       ; save coordinates every 1 ns
nstvout                  = 250000       ; save velocities every 1 ns
compressed-x-precision   = 1000

; This selects the subset of atoms for the .xtc file. You can
; select multiple groups. By default all atoms will be written.
compressed-x-grps        = non-Water

; Selection of energy groups
energygrps               = System
"""


        self.mdp_text +=  """; Neighborsearching and short-range nonbonded interactions
nstlist                  = 10
ns_type                  = grid
pbc                      = xyz
rlist                    = 0.9

; Electrostatics
cutoff-scheme            = verlet
coulombtype              = PME
rcoulomb                 = 0.9

; van der Waals
vdw-type                 = Cut-off
vdw-modifier             = Potential-switch
rvdw-switch              = 0.89      ;    0.9
rvdw                     = 0.9

; Apply long range dispersion corrections for Energy and Pressure 
; YES -- we're doing NPT
DispCorr                 = EnerPres

fourierspacing           = 0.10
pme_order                = 4
ewald_rtol               = 1e-6
ewald_geometry           = 3d
epsilon_surface          = 0


; Temperature coupling
tcoupl                   = v-rescale
nsttcouple               = 1
tc_grps                  = System
tau_t                    = 0.5
ref_t                    = {temperature}

; Pressure coupling is on for NPT - we're doing NVT.  Berendsen is recommended for use with position restraints
pcoupl                   = Berendsen
pcoupltype               = isotropic                   ; uniform scaling of x-y-z box vectors
tau_p                    = 2.0                         ; time constant, in ps
ref_p                    = 1.0                         ; reference pressure (in bar)
compressibility          = 4.5e-5                      ; isothermal compressibility, bar^-1
refcoord_scaling         = com


; velocity generation
gen_vel                  = yes
gen-temp                 = {temperature}
gen-seed                 = {randseed} ; need to randomize the seed each time.

; options for bonds
constraints              = h-bonds  ; we only have C-H bonds here
; Type of constraint algorithm
constraint-algorithm     = lincs
; Highest order in the expansion of the constraint coupling matrix
lincs-order              = 12
lincs-iter               = 2


; FREE ENERGY CONTROL OPTIONS =
free-energy   	        = expanded
calc-lambda-neighbors 	= -1
sc-alpha 		= 0.5    ;     0.5 
sc-power 		= 1      ;     keep this at 1 
sc-sigma 	        = 0.3    ;     0.5
couple-moltype 		= {couple_moltype}  ; ligand mol type
couple-lambda0 		= vdw-q
couple-lambda1 		= none
couple-intramol 	= yes
init-lambda-state	= {init_lambda_state}

nstexpanded             = {nstexpanded}  
nstdhdl                 = {nstexpanded}  ; dhdl snapshot freq  <-- MUST be same as nstexpanded
dhdl-print-energy 	= total
nst-transition-matrix 	= 250000

lmc-seed                = {randseed} ; should be randomized
lmc-gibbsdelta          = 1 ; transition only between nearest neighbors, -1 for all possible i->j
symmetrized-transition-matrix = yes

lmc-stats                       = wang-landau
lmc-move                        = metropolized-gibbs
lmc-weights-equil               = wl-delta
weight-equil-wl-delta           = 0.00001
init-wl-delta                   = {wl_increment_in_kT}   ; in units kT -  MRS had 10.0 at first
separate-dhdl-file              = yes
wl-scale                        = 0.8
wl-ratio                        = 0.7

coul-lambdas         = {coul_lambdas_string}
vdw-lambdas          = {vdw_lambdas_string}
fep-lambdas          = {fep_lambdas_string}
restraint-lambdas    = {rest_lambdas_string}
init-lambda-weights  = {init_lambda_weights}

"""

        if not self.ligand_only:
            self.mdp_text +=  """; 
; pulling parameters
pull                     = yes
pull-ngroups             = 2
pull-ncoords             = 1
pull-group1-name         = {pull_group1_name}
pull-group2-name         = {pull_group2_name}
pull-coord1-geometry            = {pull_geometry}     
pull-coord1-groups       = 1 2
pull-coord1-dim                 = Y Y Y
pull-coord1-rate         = 0.00
pull-coord1-k            = {pull_coord1_k}
pull-coord1-start               = no
pull-coord1-init         = {pull_coord1_init}
pull-nstxout             = 250   ; 1 ps
pull-nstfout             = 250   ; 1 ps
"""


        # format the mdpfile text!
        self.mdp_text = self.mdp_text.format(
            dt                  = self.dt,
            nsteps              = self.nsteps,
            nstexpanded         = self.nstexpanded, 
            couple_moltype      = self.couple_moltype, 
            pull_group1_name    = self.pull_group1_name, 
            pull_group2_name    = self.pull_group2_name,
            pull_geometry       = self.pull_geometry,
            pull_coord1_k       = self.pull_coord1_k,
            pull_coord1_init    = self.pull_coord1_init,
            fep_lambdas_string  = self.fep_lambdas_string,
            coul_lambdas_string = self.coul_lambdas_string,
            vdw_lambdas_string  = self.vdw_lambdas_string,
            rest_lambdas_string = self.rest_lambdas_string,
            init_lambda_state   = self.init_lambda_state,
            init_lambda_weights = self.init_lambda_weights_string,
            wl_increment_in_kT  = self.wl_increment_in_kT,
            randseed            = self.randseed,
            temperature         = self.temperature )

        return self.header_desc + self.mdp_text
    
    def write_to_filename(self, filename):
        """Writes the ee mdpfile to specified filename."""

        fout = open(filename, 'w')
        fout.write( self.generate_mdp_text() )         
        fout.close()


