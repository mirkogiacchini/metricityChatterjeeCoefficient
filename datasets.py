from utils import *
import tqdm
from ucimlrepo import fetch_ucirepo 
import os
import json

class Dataset():
    def __init__(self, name, matrix_values):
        self.values = matrix_values
        self.name = name

    def num_rand_vars(self):
        return len(self.values[0])
    
    def num_instances(self):
        return len(self.values)

    def random_sample(self, rnd):
        return self.values[rnd.randint(0, len(self.values)-1)]

    def generate_samples(self, rnd, num_samples):
        rv_samples = [[] for _ in range(self.num_rand_vars())]
        for _ in range(num_samples):
            sample = self.random_sample(rnd)
            for i in range(len(sample)):
                rv_samples[i].append(sample[i])
        return rv_samples
    
    def compute_all_pairs_chatterjee(self, rnd, num_samples, show_tqdm=True):
        rv_samples = self.generate_samples(rnd, num_samples)
        corr_coeff = {}
        for i in tqdm.tqdm(range(self.num_rand_vars()), disable=not show_tqdm):
            for j in range(self.num_rand_vars()):
                if i == j:
                    continue
                coeff = chatterjee(rv_samples[i], rv_samples[j], rnd)
                corr_coeff[(i,j)] = coeff
        return corr_coeff

def uci_dataset(repo_id):
    path = f'data/uci/{repo_id}'
    if os.path.exists(path):
        with open(path, 'r') as f:
            values = json.load(f)
    else:
        os.makedirs('data/uci/', exist_ok=True)
        ds = fetch_ucirepo(id=repo_id) 
    
        # data (as pandas dataframes) 
        X = ds.data.features 
        
        vars = ds.variables.values.tolist()
        good_cols = []
        for v in vars:
            if v[1] == 'Feature' and v[2] == 'Continuous' and v[6] == 'no':
                good_cols.append(v[0])
        if len(good_cols) > 100:
            print(f'Skipping {repo_id}, #cols:{len(good_cols)}')
            values = [[]]
        else:
            X = X[good_cols]
            X = X.dropna(axis=1) #remove columns with nan
            values = X.values.tolist()
            try:
                for i in range(len(values)):
                    for j in range(len(values[i])):
                        values[i][j] = float(values[i][j])
            except:
                values = [[]]
        with open(path, 'w') as f:
            json.dump(values, f)
    return Dataset(f'uci{repo_id}', values)