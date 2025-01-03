import os, sys
import numpy as np


class MixingTime(object):
    """A class to perform operations related to EE mixing times."""

    def __init__(self, P_acc, K):
        """Initializer for the MixingTime class."""

        self.P_acc = P_acc
        self.K = K

        self.T = self.transition_matrix()
        self.evals = self.analytical_evals()
        self.t_k = self.implied_timescales()


    def transition_matrix(self, P_acc=None, K=None):
        """Returns a tridiagonal transition matrix that corresponds to 1D
        diffusion along a set of K intermediates with transition rate P_acc:

	| b+a  a         ...                |  
        | a    b    a                       |
        |      a    b    a                  |

        | ...      ...        a    b     a  |
        |                          a    b+a |

        where a = P_acc, b = 1 - 2P_acc.


        INPUTS
        P_acc - the acceptance probability of transitioning to neighboring states.
        K     - the number of thermodynamic states

        OUTPUTS
        T     - the K x K transition matrix as np.array() object
        """

        if P_acc == None:
            P_acc = self.P_acc
        if K == None:
            K = self.K

        a = P_acc
        b = 1.0 - (2.0* P_acc)

        # fill the diagonal with b
        T =  np.diag( b*np.ones(K)) 

        # fill upper diagonal with a
        T += a*np.eye(K, k=1) 

        # fill lower diagonal with a
        T += a*np.eye(K, k=-1)

        # add a to the ends
        T[0,0] += a
        T[K-1, K-1] += a

        return T



    def analytical_evals(self, P_acc=None, K=None):
        """Returns a sorted list of evals (from largest to smallest) of a tridiagonal transition
        matrix that corresponds to 1D diffusion along a set of K intermediates with transition rate P_acc,
        according to the formula


        $ \mu_k = 1 - 2P_acc[1 - \cos \frac{(k-1)\pi}{K} $

        see Theorem 5 of Yueh, Wen-Chyuan (2005)

        REFERENCE 
        Yueh, W. Applied Mathematics E-Notes, 5(2005), 66-74 􏰀c ISSN 1607-2510.
        Available free at mirror sites of http://www.math.nthu.edu.tw/∼amen/
        """

        if P_acc == None:
            P_acc = self.P_acc
        if K == None:
            K = self.K

        k_values = np.arange(K)

        mu_k = 1.0 - 2.0*P_acc*(1.0 - np.cos( k_values  * np.pi / K ))

        return mu_k



    def implied_timescales(self, P_acc=None, K=None, kmax=10):
        """Returns a list of implied timescales t_k, k=2,....t_min(K, kmax), where 
        the implied timescales are

        $ t_k =  \frac{-1}{\ln \mu_k}  $

        The implied timescales are reported in units of steps (or transition matrix lagtimes).
        """

        if P_acc == None:
            P_acc = self.P_acc
        if K == None:
            K = self.K

        # If either parameter is re-specified, rebuild the transition matrix
        if (P_acc == None) or (K == None):
            T = self.transition_matrix(P_acc=P_acc, K=K)	
            evals = self.analytical_evals(P_acc=P_acc, K=K)
        else:
            T = self.T
            evals = self.evals

        t_k = np.array( [-1.0/ np.log(evals[k]) for k in range(1, min(K, kmax)) ] )
        return t_k
 


if __name__ == '__main__':


    P_acc = 0.1
    K = 5

    m = MixingTime(P_acc, K)

    print('--- Performing tests of the MixingTime class ---')
    print(f'K = {K}')
    print(f'P_acc = {P_acc}')
    print()
    print('Transition matrix:', m.T)


    print('\nDo the analytical eigenvalues match the numerically computed values?')

    evals, evecs = np.linalg.eig(m.T)
    Ind = np.argsort(-evals)
    print('Numerical evals from np.linalg.eig():', evals[Ind])

    print('Analytic evals:', m.evals)

    print('\nImplied timescales:', m.t_k)



    


