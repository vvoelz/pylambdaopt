import glob, re
from EEAnalysis import EEAnalysis
import pandas as pd
data = []
#for dir in ['L/pre-opt','L/post-opt']:
for dir in ['RL/pre-opt','RL/post-opt','RL/post-opt-opt','RL/post-opt-opt-opt']:
    for run in [0,1,3,4,5,6,7]:
#        for j in [1000,5000,30000]:
            try:
                e = EEAnalysis( RL_directory=f'{dir}/RUN{run}') #,
#                  L_directory=f'L/post-opt/RUN{run}')
                e.scrape_log()
#                results = e.analyze()
#                data.append([run] + results)
            except Exception as e:
                print(f'RUN{run} failed: {e}')

#df = pd.DataFrame(data, columns=['run','dG1','dG2','dG3',
#  'dG_binding','logKd','dG2_sigma','dG3_sigma','dG_bind_sigma','logKd_sigma',
#  'RL_increment','L_increment','RL_ns','L_ns'])
#df.to_csv('opt2_results.csv')    
