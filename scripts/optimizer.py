import os, sys
import numpy as np

import scipy
from scipy import *

from mixing import *


class Optimizer(object):
    """A class to optimize alchemical intermediates by spacing them equally in thermodynamic length."""


    def __init__(self):
        """Initializes the class."""

        self.L_spl    = None    # to be filled with a cubic spline function for L(alpha)
        self.L_spl_1d = None    # to be filled with a function for the 1st derivative L'(alpha)



    def create_spline(self, alpha_values, L_values):
        """Creates and returns a differential cubic spline function fit to a provided set of
           observed alpha_k and thermodynamic lengths $\mathcal{L}(0,alpha_k),

        INPUTS
	alpha_values - an observed list of alpha values
        L_values     - an observed list of L_values (the numbers of values must match) 

        OUTPUT
        L_spl    - A cubic spline function that can be called like L_spl(alpha)
        L_spl_1d - The first derivative of the spline, L_spl_1d = y_spl.derivative(n=1) 
	""" 
        
        assert len(alpha_values) == len(L_values)

        from scipy.interpolate import UnivariateSpline
        from scipy.interpolate import interp1d

        self.L_spl = UnivariateSpline(alpha_values, L_values, s=0, k=3)
        self.L_spl_1d = self.L_spl.derivative(n=1) 

        return 


    def optimize_alphas(self, alphas, nsteps=100000, tol=1e-7, gamma=1e-5, max_del=0.0001, print_every=2000):
        """Given a set of alphas, optimize

        NOTE: These set of alphas could be different from the set a_k that the spline was fit to.  In this way
              we can space arbitrary numbers of alpha values.


        INPUTS
        alphas  -  a set of K alpha values to be optimized, with alphas[0] and alphas[K-1] treated as fixed endpoints

        PARAMETERS
        nsteps  -  the number of minimization steps.  Default: 100000
        tol     -  the tolerance to stop at if the alphas don't change within.  Default: 1e-7
        gamma   -  the gradient descent step size. Default: 1e-5
        max_del -  the maximum allowed change in alpha as a step size.  Default: 0.0001
        print_every - frequency (number of steps) to output progess. Default: 2000

        RETURNS
        new_alphas  - the final set of optimized alpha values
        traj_alphas - a (K, nsteps) np.array storing the trajectory of the minimization for all alpha_k 
        """


        assert (self.L_spl != None) and (self.L_spl_1d != None), \
            "The cubic spline functions need to be defined before running this routine! Please run Optimizer.create_spline() first,"

    
        K = len(alphas)
        print('K = ', alphas)

        old_alphas = np.array(alphas)
        traj_alphas = np.zeros( (K, nsteps) )

        for step in range(nsteps):

            # store the trajectory of alphas
            traj_alphas[:,step] = old_alphas
            if step%print_every == 0:
                print('step', step, old_alphas)
    
            # perform a steepest descent step
            new_alphas = np.zeros( old_alphas.shape )
            del_alphas = np.zeros( old_alphas.shape )

            del_alphas[0] = 0.0    # fix the alphas[0] endpoint
            del_alphas[K-1] = 0.0  # fix the alpha[K-1] endpoint
    
            if False:  # calculate del_alphas using a for loop (this is too SLOW!) 
                for i in range(1, (K-1)):
                    del_alphas[i] = -1.0*gamma*2.0*self.L_spl_1d(old_alphas[i])*( 2.0*self.L_spl(old_alphas[i]) - self.L_spl(old_alphas[i-1]) - self.L_spl(old_alphas[i+1]))

            else:   # calculated del_alphas as a vector operation (FASTER! - do this instead)
                L_all = self.L_spl(old_alphas)
                Lh, Li, Lj = L_all[0:K-2], L_all[1:K-1], L_all[2:K] 
                del_alphas[1:K-1] = -1.0*gamma*2.0*self.L_spl_1d(old_alphas[1:K-1])*( 2.0*Li - Lh - Lj)

            if abs(np.max(del_alphas)) > max_del:
                del_alphas[1:K-1] = del_alphas[1:K-1]*max_del/np.max(del_alphas)
            new_alphas = old_alphas + del_alphas
            
            # record the average change in the alphas 
            del_alphas = np.abs(old_alphas - new_alphas).mean()
            if step % print_every == 0:
                print('step', step, 'del_alphas', del_alphas)
            if del_alphas < tol:
                print('Tolerance has been reached: del_alphas =', del_alphas, '< tol =', tol)
                break
        
            old_alphas = new_alphas

        return new_alphas, traj_alphas[:,0:step]
        


    def create_phi(self, labels=['rest', 'coul', 'vdw'], ascending=[True, True, True] ):
        """Returns a path function phi(\alpha) = \vec{\lambda} that can be used to convert \alpha_k values to sets of \vec{lambda_k}.

        NOTE: The mapping assumes that \alpha is in in the range [0, nlambdas], and that 
              each lambda is turned "on" (or "off") in succession in each interval [0,1], [1, 2], etc.... 

              In each interval:
                  if ascending == True: \lambda =0 \rightarrow 1; else: \lambda = 1 \rightarrow 0.""" 


        assert len(labels) == len(ascending) 

        nlambdas = len(labels)

        def phi(alpha):
            """Return the value of mapping function phi(\alpha) \rightarrow \vec{lambda}."""

            if np.isscalar(alpha):
                result = np.zeros(nlambdas) 
                for i in range(nlambdas):
                    if ascending[i]:
                        result[i] = min( max(0, alpha - i), 1.0)
                    else:
                        result[i] = min( max(0, 1.0 - alpha + i), 1.0)
                return result

            else:
                N = len(alpha)
                result = np.zeros((N, nlambdas))
                for k in range(N):
                    for i in range(nlambdas):
                        if ascending[i]:
                            result[k,i] = min( max(0, alpha[k] - i), 1.0)
                        else:
                            result[k,i] = min( max(0, 1.0 - alpha[k] + i), 1.0)
                return result


        return phi

    def optimize_K(self, alpha_1, alpha_K, min_K=5, max_K=200):
        """Given an already-fit spline, adjust the number K of intermediates alpha_k between
        two endpoints alpha_1 and alpha_K to minimize the mixing time.

        INPUTS
        alpha_1 - the left endpoint of the alpha_k
        alpha_K - the right endpoint of the alpha_k

        PARAMETERS
        min_K   - the minimum value of K to consider. Default: 5
        max_K   - the maximum value of K to consider. Default: 200  

        

        """

        assert (self.L_spl != None) and (self.L_spl_1d != None), \
            "The cubic spline functions need to be defined before running this routine! Please run Optimizer.create_spline() first,"

        # estimate the total thermodynamic length between the endpoints
        L_tot = self.L_spl(alpha_K) - self.L_spl(alpha_1)

        # for a range of K, calculate the mixing times t2
        K_values = np.arange(min_K, max_K+1)
        P_acc_values = []
        t2_values = []
        for K in K_values:

            L = L_tot/(K-1)
            P_acc = np.exp(-L**2 / 2) 
            P_acc_values.append(P_acc)

            m = MixingTime(P_acc, K)   # initializing this object will compute the transition matrix T and implied timescales t_k
            t2_values.append( m.t_k[1] ) # the second implied timescale is t2 (the first is t1 = 1.)  


        # determine the optimal K = argmin(t2)
        t2_values = np.array(t2_values)
        P_acc_values = np.array(P_acc_values)

        Ind = np.isfinite(t2_values)
        finite_t2_values = t2_values[Ind]
        finite_K_values = K_values[Ind]

        optimal_K = finite_K_values[np.argmin(finite_t2_values)]

        return optimal_K, K_values, P_acc_values, t2_values



##################################

if __name__ == '__main__':

    import numpy as np
    
    o = Optimizer()
    phi =  o.create_phi()
    for alpha in np.arange(0,3,0.1):
        print( phi(alpha) )


    
    
