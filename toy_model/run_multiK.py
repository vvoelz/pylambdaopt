from sampler import *
import numpy as np

ntrials = 5

for K in range(10, 100, 5):

    for trial in range(ntrials):

        lambdas = np.linspace(0., 1.0, K)
        s = ToySampler(lambdas)

        s.sample(nsteps=10000000, print_every=50000, traj_every=100)
        s.save_traj_data(f'test10M_K{K}_uniform_trial{trial}')

        del(s)


