import os, sys, copy
import numpy as np

from datetime import datetime


class ToySampler(object):
    """A class to set up and perform Expanded Ensemble sampling for a toy model
    of 1-D harmonic potentials.

        U(x)  = (k_f/2) * (x - \lambda*x_0)**2  

    """

    def __init__(self, lambdas, k_f=10.0):
        """Initialize the class.

        INPUTS
        lambdas - a set of lambda values provided in a np.array object

        PARAMETERS
        k_f - the force constant for the harmonic well, in units k_BT.
        """

        # Harmonic potential settings
        self.k_f = k_f
        self.x0  = 10.0
        self.lambdas = lambdas   #  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.65, 0.8, 0.9, 1.0])
        self.K = len(self.lambdas)

        # Wang-Landau settings
        self.f_k = np.zeros(self.K)  # bias energies are -f_k
        self.h_k   = np.zeros(self.K)  # histogram counts

        self.wl_increment = 10.0   # in kT
        self.wl_scaling   = 0.8
        self.flatness     = 0.7    # histograms are sufficinetly flat if eta < h_k/avg(h) < eta^{-1}
        self.ee_freq = 10  # frequency to attempt EE transitions

        # Trajectory data
        self.x_traj = []
        self.k_traj = []
        self.step_traj = []
        self.wl_increment_traj = []
        self.f_k_traj = []
        self.u_k_traj = []

        # MCMC statisticss
        self.attempted = np.zeros((self.K, self.K))
        self.accepted  = np.zeros((self.K, self.K))
        self.acc_probs = np.zeros((self.K, self.K))

 


    def u_k(self, x):
        """Returns the reduced energies for all k=1...K ensembles.

        INPUTS
        x - a scalar value
        """

        return self.k_f/2.0 * (x - self.lambdas*self.x0)**2


    def sample(self, nsteps=200000, x_init=0.0, dx=0.1, k_init=0, print_every=10000, traj_every=100):
        """Perform EE sampling.

        INPUTS


        PARAMETERS
        nsteps  - the number of steps in the simulation. Default: 200000
        x_init  - the initial position. Default: 0.
        dx      - the Gaussian step size. Default: 0.1
        k_init  - the initial thermodynamic state
        print_every - frequency in steps to print status. Default: 10000
        traj_every  - frequency in steps to save trajectory snapshots. Default: 100

        RETURNS

        """ 

        # initialize WL settings
        self.wl_increment = 10.0   # in kT


        # initialize the sampling trajectories
        self.x_traj = []
        self.k_traj = []
        self.step_traj = []
        self.wl_increment_traj = []
        self.f_k_traj = []
        self.u_k_traj = []

        # MCMC stats
        self.attempted = np.zeros((self.K, self.K))
        self.accepted  = np.zeros((self.K, self.K)) 
        self.acc_probs = np.zeros((self.K, self.K))

        # timing 
        startTime = datetime.now()


        k = k_init 
        x = x_init
        u = self.u_k(x)[k] 
   
        # Perform  sampling
        for step in range(nsteps):

            # propose a MC move
            x_new = x + dx*np.random.randn()
            u_new = self.u_k(x_new)[k]
    
            # and accept it according to the metropolis criterion
            accept = False
            P_accept = min(1., np.exp(-(u_new - u)))
            if np.random.rand() < P_accept:
                accept = True
    
            if accept:
                x = x_new
                u = u_new
    
            # attempt EE transitions
            if (step % self.ee_freq == 0):
        
                self.f_k[k] -= self.wl_increment
                self.h_k[k] += 1.0
        
                # reset the bias so that min(f_k)
                self.f_k -= np.min(self.f_k)
        
                # attempt a move to a neighboring ensemble
                wl_accept = False
                if np.random.rand() < 0.5:
                    l = k+1
                else:
                    l = k-1
        
                if (l >= 0) and (l < self.K):
                    u_k = self.u_k(x)
                    P_wl_accept = min(1., np.exp(-(u_k[l] - self.f_k[l] - u_k[k] + self.f_k[k])))
                    self.attempted[k,l] += 1
                    if np.random.rand() < P_wl_accept:
                        wl_accept = True    
                    if wl_accept:
                        self.accepted[k,l] += 1
                        k = l   # move the thermodynamic index

            
                # check if the histogram is flat enough
                mean_h  = self.h_k.mean()
                Ind1 = (self.h_k/mean_h > self.flatness) 
                Ind2 = (self.h_k/mean_h < 1.0/self.flatness)
                if sum((Ind1*Ind2).astype(int)) == self.K:
                    self.wl_increment *= self.wl_scaling
                    self.h_k = np.zeros(self.K)  # reset the histogram counts

            # print a status report
            if step % print_every == 0:
                print()
                print(f'step {step} of {nsteps}: x {x:2.6f}, k {k} wl_increment = {self.wl_increment:1.6f} kT')
                print()
                print('# ensemble\thistogram\tg (kT)')
                for i in range(self.K):
                    outstr = '%8d\t%8d\t%3.4f'%(i, self.h_k[i], self.f_k[i])
                    if k == i:
                        outstr += ' <<'
                    print(outstr)

            # store the sample in trajectory 
            if step % traj_every == 0:
                self.x_traj.append(x)
                self.k_traj.append(k)
                self.step_traj.append(step)
                self.f_k_traj.append(copy.copy(self.f_k))
                self.wl_increment_traj.append(copy.copy(self.wl_increment))
                self.u_k_traj.append(self.u_k(x) - self.u_k(x)[k])
        

        self.acc_probs = self.accepted/(self.attempted + 0.0001)  # avoid division by zero


        
    def save_traj_data(self, outdir):
        """Saves all th trajectory data to a set of readable files in the named directory."""

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # make sure the trajectory lists are all the same length so we can store them in an array
        assert len(self.x_traj) == len(self.step_traj)
        assert len(self.k_traj) == len(self.step_traj)
        assert len(self.f_k_traj) == len(self.step_traj)
        assert len(self.wl_increment_traj) == len(self.step_traj)
        assert len(self.u_k_traj) == len(self.step_traj)

        N = len(self.step_traj)

        # lambda_values
        lambda_values_file = os.path.join(outdir, 'lambda_values.txt')
        np.savetxt(lambda_values_file, self.lambdas, fmt=['%1.6f'], delimiter='\t', header='#lambda_k')
        print(f'Wrote: {lambda_values_file}')

        # coord trajectory 
        coord_traj = np.zeros( (N, 3))
        coord_traj[:,0] = np.array(self.step_traj) 
        coord_traj[:,1] = np.array(self.k_traj) 
        coord_traj[:,2] = np.array(self.x_traj) 

        coord_trajfile = os.path.join(outdir, 'coord_traj.txt')
        np.savetxt(coord_trajfile, coord_traj, fmt=['%d','%d','%2.6f'], delimiter='\t', header='#step\tk\tx')
        print(f'Wrote: {coord_trajfile}')

        # wl_increment trajectory 
        wl_increment_traj = np.zeros( (N, 3))
        wl_increment_traj[:,0] = np.array(self.step_traj)
        wl_increment_traj[:,1] = np.array(self.k_traj)
        wl_increment_traj[:,2] = np.array(self.wl_increment_traj)

        wl_increment_trajfile = os.path.join(outdir, 'wl_increment_traj.txt')
        np.savetxt(wl_increment_trajfile, wl_increment_traj, fmt=['%d','%d','%1.6f'], delimiter='\t', header='#step\tk\twl_increment(kT)')
        print(f'Wrote: {wl_increment_trajfile}')



        # f_k trajectory 
        f_k_traj = np.zeros( (N, (self.K)+2))
        f_k_traj[:,0] = np.array(self.step_traj)
        f_k_traj[:,1] = np.array(self.k_traj)
        f_k_traj[:,2:] = np.array(self.f_k_traj)

        f_k_trajfile = os.path.join(outdir, 'f_k_traj.txt')
        np.savetxt(f_k_trajfile, f_k_traj, fmt=(['%d','%d']+self.K*['%2.6f']), delimiter='\t', header='#step\tk\tf_k')
        print(f'Wrote: {f_k_trajfile}')

        # u_k trajectory 
        u_k_traj = np.zeros( (N, (self.K)+2))
        u_k_traj[:,0] = np.array(self.step_traj)
        u_k_traj[:,1] = np.array(self.k_traj)
        u_k_traj[:,2:] = np.array(self.u_k_traj)

        u_k_trajfile = os.path.join(outdir, 'u_k_traj.txt')
        np.savetxt(u_k_trajfile, u_k_traj, fmt=(['%d','%d']+self.K*['%1.6e']), delimiter='\t', header='#step\tk\tu_k')
        print(f'Wrote: {u_k_trajfile}')


        # transition matrix
        transition_matrix_file = os.path.join(outdir, 'transition_matrix.txt')
        np.savetxt(transition_matrix_file, self.acc_probs, fmt='%1.6e', delimiter='\t')
        print(f'Wrote: {transition_matrix_file}')



if __name__ == '__main__':

    lambdas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.65, 0.8, 0.9, 1.0])
    s = ToySampler(lambdas)
    s.sample(nsteps=2000000, print_every=1000, traj_every=100)

    s.save_traj_data('testout')



