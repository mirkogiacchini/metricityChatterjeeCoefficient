from utils import *
from datasets import *
import random, os, json

def run_experiment(dataset, seed, res_dir, num_samples):
    print(f'============ {dataset.name} {seed}')
    rand = random.Random(seed)
    chatterjee_corr_coeff = dataset.compute_all_pairs_chatterjee(rand, num_samples)

    os.makedirs(res_dir+f'{dataset.name}/', exist_ok=True)
    out_path = res_dir+f'{dataset.name}/{seed}.json'
    with open(out_path, 'w') as f:
        json.dump(dict_str_format(chatterjee_corr_coeff), f)

if __name__ == '__main__':
    res_dir = 'results/'
    
    seeds = [42, 100, 200, 300, 400]

    uci_ds_list = get_uci_datasets_id()
    for seed in seeds:
        for uci_ds in tqdm.tqdm(uci_ds_list):
            dataset = uci_dataset(uci_ds)
            if dataset.num_rand_vars() >= 3:
                run_experiment(dataset, seed, res_dir, 50000)
        
