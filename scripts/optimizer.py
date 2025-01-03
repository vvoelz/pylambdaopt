import os, sys

import scipy
from scipy import *


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
        self.L_spl_1d = L_spl.derivative(n=1) 

        return 


    def optimize_alphas(alphas, nsteps=100000, tol=1e-7, gamma=1e-5, max_del=0.0001, print_every=2000):
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

        return new_alphas, traj_alphas
        


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
            result = np.zeros(nlambdas) 
            for i in range(nlambdas):
                if ascending[i]:
                    result[i] = min( max(0, alpha - i), 1.0)
                else:
                    result[i] = min( max(0, 1.0 - alpha + i), 1.0)
            return result

        return phi


##################################

if __name__ == '__main__':

    import numpy as np
    
    o = Optimizer()
    phi =  o.create_phi()
    for alpha in np.arange(0,3,0.1):
        print( phi(alpha) )


    
    
