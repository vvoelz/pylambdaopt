import numpy as np
from sampler import *

lambdas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.65, 0.77, 0.88, 1.0])

s = ToySampler(lambdas)

s.sample(nsteps=10000000, print_every=10000, traj_every=100)
        
s.save_traj_data('test10M_unoptimized')
