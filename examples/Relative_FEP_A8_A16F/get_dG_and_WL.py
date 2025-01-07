import numpy as np
import matplotlib.pyplot as plt
import os, sys, math
import pandas as pd
import datetime
from natsort import natsorted as ns
from matplotlib import colors
import seaborn as sns
from matplotlib.patches import Rectangle
from multiprocessing import Pool,cpu_count

"""
Usage: python get_dG_and_WL.py filename number_of_lambdas
"""
filename = sys.argv[1]
lambdas = sys.argv[2]
temp_name = filename.split('.')[0]
name = '_'.join(temp_name.split('_')[-1])

def get_dG_and_WL(filename):
   all_log_WL_incrementor, all_G_values =[],[]
   fn = open(f'{filename}', 'r', encoding='latin-1')
   lines = fn.readlines()
   fn.close()
   for l in range(len(lines)):
          line = lines[l].split()
          if len(line) !=0 and line[0] == 'Wang-Landau':
              all_log_WL_incrementor.append(math.log10(float(line[3])))
              #print(lines[l+46])
              all_G_values.append(float(lines[l+lambdas+1].split()[3]))
   np.save(f'log_WL_{name}.npy', all_log_WL_incrementor)
   np.save(f'dGs_{name}.npy', np.array(all_G_values))
if __name__ == "__main__":
    get_dG_and_WL(filename)
