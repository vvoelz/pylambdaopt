from sampler import *

# From test10M_unoptimized_opt:

optimized_lambdas = np.loadtxt('test10M_unoptimized_opt/optimized_lambdas.txt')

s = ToySampler(optimized_lambdas)

s.sample(nsteps=10000000, print_every=10000, traj_every=100)
        
s.save_traj_data('test10M_optimized')
