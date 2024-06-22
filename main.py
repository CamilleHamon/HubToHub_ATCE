#%%
import pandas as pd
from pathlib import Path
import cvxpy as cp
from max_flow import Edge, Node
from itertools import combinations
from tqdm import tqdm
import pickle

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

#%% Extract list of borders and bidding zones
all_borders = []
for b in atcs.columns:
    from_bz, to_bz = b.split('-')
    # Check if we encountered the border in the other direction
    b_other_dir = f'{to_bz}-{from_bz}'
    if b_other_dir not in all_borders:
        all_borders += [b]
all_bidding_zones = set([bz for b in all_borders for bz in b.split('-')])
all_bz_pairs = set(combinations(all_bidding_zones,2))

#%% Solve the max-flow problem for each mtu and combination of bidding zones
# Construct the edges and solve problem for each mtu
nb_mtu = atcs.shape[0]
nb_bz_pairs = len(all_bz_pairs)
nb_iter = nb_mtu*nb_bz_pairs

with tqdm(total=nb_iter) as pbar:
    all_results = {}
    for mtu, act_mtu in atcs.iterrows():
        # Solve max-flow problem for each bidding zone pair
        for source,sink in all_bz_pairs:
            pbar.set_description(f'MTU:{mtu}, bidding zone pair: {source}>{sink}')
            nodes = {bz: Node(name=bz) for bz in all_bidding_zones} 
            edges = {}
            for b, atc_b in act_mtu.items():
                from_bz,to_bz = b.split('-')
                # We add an edge only if the other direction doesn't already exist
                # since edges are bi-directional
                b_other_dir = f'{to_bz}-{from_bz}'
                if b_other_dir not in list(edges.keys()):
                    capacity_forward = atc_b
                    capacity_reverse = -act_mtu[b_other_dir]
                    new_edge = Edge(capacity_forward,capacity_reverse,nodes[from_bz],nodes[to_bz])
                    edges[b] = new_edge
            # Set the source and sink bidding zone
            nodes[source].accumulation = cp.Variable()
            nodes[sink].accumulation = cp.Variable()
            # Build constraints
            constraints = []
            for o in list(nodes.values()) + list(edges.values()):
                constraints += o.constraints()
            # Objective function: maximize trade to sink from source
            p = cp.Problem(cp.Maximize(nodes[sink].accumulation), constraints)
            # Solve
            # print(f'solve max-flow problem for mtu {mtu} from {source} to {sink}')
            results = p.solve(solver='CLARABEL')
            # print('Results:')
            # print(f'Trading capacity between {source} and {sink}: {results} MWh')
            # print(f'Corresponding trading flows on borders:')
            # for e in edges.values():
            #     if e.flow.value != 0:
            #         print(e)
            all_results[(mtu,source,sink)] = {
                'Max trading capacity': results,
                'Cross-border trading flows': edges
            }
            pbar.update()

#%% Save hub-to-hub capacities in csv
h2h_caps = {k: {'H2H capacity': v['Max trading capacity']} for k,v in all_results.items()}
h2h_caps = pd.DataFrame.from_dict(h2h_caps,orient='index').rename_axis(['MTU','Source','Sink'])
result_folder = Path('Results') / week_folder.name
result_folder.mkdir(parents=True, exist_ok=True)
result_file = 'Hub-to-hub_ID_capacities.csv'
h2h_caps.reset_index().to_csv(result_folder / result_file,index=False)
# %% Save cross-border flows in csv
edge_results = {}
for k,v in all_results.items():
    for e in v['Cross-border trading flows'].values():
        if abs(e.flow.value) > 1e-5:
            key = k + (e.from_node,e.to_node)
            edge_results[key] = {'Flow': e.flow.value}
edge_results = pd.DataFrame.from_dict(edge_results,orient='index').rename_axis(['MTU','Source','Sink','From','To'])
edge_cap_file = 'Hub-to-hub_cb_flows.csv'
edge_results.reset_index().to_csv(result_folder / edge_cap_file,index=False)


# %%
