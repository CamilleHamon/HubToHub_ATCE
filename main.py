#%%
import pandas as pd
from pathlib import Path
import cvxpy as cp
from max_flow import Edge, Node

pd.options.mode.copy_on_write = True

data_folder = Path('Data')
week_folder = '2024w20_public/'
week_folder = data_folder / week_folder
csv_files = week_folder.glob('*.csv')
all_weekly_res = []
for f in csv_files:
    data = pd.read_csv(f,header=[0,1])
    data[('MTU','MTU')] = pd.to_datetime(data[('MTU','MTU')])
    data = data.set_index(('MTU','MTU'))
    data.index.name = 'MTU'
    data = data.drop(('Backup','Backup'),axis=1)
    data.columns.names = ['Border','Quantity']
    # drop unnamed columns
    cols_to_remove = []
    for c in data.columns:
        if ('Unnamed' in c[0]) or ('Unnamed' in c[1]):
            cols_to_remove.append(c)
    data = data.drop(cols_to_remove,axis=1)
    all_weekly_res.append(data)
atcs = pd.concat(all_weekly_res).sort_index().swaplevel(axis=1)['ATC']

#  remove NO2A, SE3A, SE3_AC, SE4_AC, SE3_SWL, SE4_SWL
bzs_to_remove = ['NO2A', 'SE3A', 'SE3_AC', 'SE4_AC', 'SE3_SWL', 'SE4_SWL']
borders_to_remove = []
for c in atcs.columns:
    if any([bz in c for bz in bzs_to_remove]):
        borders_to_remove.append(c)
atcs = atcs.drop(borders_to_remove,axis=1)

# Extract list of borders and bidding zones
all_borders = set(atcs.columns.unique(level='Border'))
all_bidding_zones = set([bz for b in all_borders for bz in b.split('-')])
# %%
