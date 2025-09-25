import json
from utils import *
from datasets import *
import math
import matplotlib.pyplot as plt
import numpy as np


def check_ti(dataset_name, aggregate='max', direct_triples=False):
    viols_per_inds = {}
    seeds = [42, 100, 200, 300, 400]
    for seed in seeds:
        path = f'results/{dataset_name}/{seed}.json'
        
        with open(path, 'r') as f:
            dd = json.load(f)   
            results = dict_to_tuple_format(dd)

        for k,v in results.items():
            results[k] = min(1, max(v, 0)) #cap, it might happen that they are out of range since we use \xi_n

        n = 1+math.sqrt(1+4*len(results))
        assert n - int(n) < 0.00001 and int(n)%2 == 0, f'{n} {int(n)} {dataset_name} {len(results)}'
        num_random_vars = int(n) // 2

        viols = []
        max_val_i = num_random_vars if direct_triples else num_random_vars-2 
        for i in range(max_val_i):
            min_val_j = 0 if direct_triples else i+1
            max_val_j = num_random_vars if direct_triples else num_random_vars-1
            for j in range(min_val_j, max_val_j):
                if i == j:
                    continue
                min_val_k = 0 if direct_triples else j+1
                for k in range(min_val_k, num_random_vars):
                    if k == i or k == j:
                        continue
                    if aggregate == 'max':
                        dij = d_sym_max(results[(i,j)], results[(j,i)])
                        djk = d_sym_max(results[(j,k)], results[(k,j)])
                        dik = d_sym_max(results[(i,k)], results[(k,i)])
                        dji = dij
                        dkj = djk
                        dki = dik
                    elif aggregate == 'min':
                        dij = d_sym_min(results[(i,j)], results[(j,i)])
                        djk = d_sym_min(results[(j,k)], results[(k,j)])
                        dik = d_sym_min(results[(i,k)], results[(k,i)])
                        dji = dij
                        dkj = djk
                        dki = dik
                    elif aggregate == 'avg':
                        dij = d_sym_avg(results[(i,j)], results[(j,i)])
                        djk = d_sym_avg(results[(j,k)], results[(k,j)])
                        dik = d_sym_avg(results[(i,k)], results[(k,i)])
                        dji = dij
                        dkj = djk
                        dki = dik 
                    elif aggregate == 'asym':
                        dij = d_asym(results[(i,j)])
                        dji = d_asym(results[(j,i)])
                        dik = d_asym(results[(i,k)])
                        dki = d_asym(results[(k,i)])
                        djk = d_asym(results[(j,k)])
                        dkj = d_asym(results[(k,j)])
                    else:
                        assert False            
                    
                    if direct_triples:
                        worst_viol = dij + djk - dik
                    else:
                        options = [dij + djk - dik, dik + dkj - dij, 
                            djk + dki - dji, dji + dik - djk,
                            dkj + dji - dki, dki + dij - dkj]
                        worst_viol = min(options)

                    if (i,j,k) not in viols_per_inds:
                        viols_per_inds[(i,j,k)] = []
                    viols_per_inds[(i, j, k)].append(worst_viol)

    viols_per_seed = {s:[] for s in seeds}
    for k,vls in viols_per_inds.items():
        for i in range(len(seeds)):
            viols_per_seed[seeds[i]].append(vls[i])
        assert len(vls) == len(seeds)
        avg_viol = sum(vls) / len(vls)
        std_viol = np.std(vls)
        viols.append((avg_viol, std_viol))
    viols = sorted(viols)
    return num_random_vars, viols, viols_per_seed

def style_plt(fontsize=None):
    plt.rc('pdf', fonttype = 42)
    plt.style.use('ggplot')
    
    plt.rcParams['axes.facecolor'] = 'white'  # Set the plot background to white
    plt.rcParams['figure.facecolor'] = 'white'  # Set the figure background to white
    plt.rcParams['grid.color'] = 'E4E4E4'  # Set grid lines to light gray
    plt.rcParams['axes.edgecolor'] = 'white'  

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('legend', fontsize=fontsize) 


def build_plot(dict_to_plot, extension, plot_name, x_label='datasets', y_label='Max Additive Violation', title=f'Max Violation across Datasets', y_lim=None, round_x_line=False, show_marker=True, figsize=(6,6), linewidth=3):
    bar_colors = {'avg': '#E69F00', 'max': '#56B4E9', 'min': '#009E73', 'asym': '#CC79A7'}
    bar_styles = {'avg': '--', 'max': '-', 'min': '-.', 'asym': ':'}
    bar_marker = {'avg': '^', 'max': 'o', 'min': '*', 'asym': '*'}
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_tight_layout(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    max_val = 0

    max_len = 0
    for v in dict_to_plot.values():
        if round_x_line:
            while v[-1][0] < 0.0001:
                v.pop()
        max_len = max(max_len, len(v))
    if round_x_line:
        max_len = max_len + (10 - (max_len%10))

    aggregate_to_name = {'max': "$d'$", 'min': "$d'_{min}$", 'avg': "$d'_{avg}$", 'asym': '$d$'}

    for aggregate,vals in dict_to_plot.items():
        avg_vals = [v[0] for v in vals]
        std_vals = [v[1] for v in vals]

        max_val = max(max_val, max(avg_vals))
        if len(avg_vals) < max_len:
            avg_vals = avg_vals + [0] * (max_len - len(avg_vals))
            std_vals = std_vals + [0] * (max_len - len(std_vals))
        avg_vals = np.array(avg_vals)
        std_vals = np.array(std_vals)
        xvals = np.arange(1,len(avg_vals)+1)
        mark_ = bar_marker[aggregate] if show_marker else ''
        ax.plot(xvals, avg_vals, label=aggregate_to_name[aggregate], color=bar_colors[aggregate], linestyle=bar_styles[aggregate], linewidth=linewidth, marker=mark_, markersize=5.5) #3.5 marker size
        ax.fill_between(xvals, avg_vals - std_vals, avg_vals + std_vals, color=bar_colors[aggregate], alpha=0.3)

        print('max std:', np.max(std_vals))
    ax.legend(frameon=False, fontsize=14)
    if y_lim is not None:
        plt.ylim(y_lim)
    else:
        plt.ylim([0, max_val+1])
    plt.title(title)
    plt.savefig(f'{plot_folder}{plot_name}{extension}', bbox_inches='tight', pad_inches=0.05)
    plt.close()

def custom_round(v):
    return math.floor(v + 0.5)

if __name__ == '__main__':
    style_plt()
    plot_folder = 'plots/'
    extension = '.pdf'
    seeds = [42, 100, 200, 300, 400]
    os.makedirs(plot_folder, exist_ok=True)

    for (direct_triples, use_all_datasets) in [(False, True), (False, False), (True, False), (True, True)]:
        if use_all_datasets:
            uci_ids = get_uci_datasets_id()
        else:
            uci_ids = get_uci_datasets_id_more_than_median()

        max_viols_per_agg = {}
        viol_distr_per_agg = {}

        for idxa, aggregate in enumerate(['max', 'min', 'avg', 'asym']):
            num_violated = 0
            num_triples_violation = {seed:{} for seed in seeds}
            max_violations = []

            for uci_id in uci_ids:
                n, viol, viols_per_seed = check_ti('uci'+str(uci_id), aggregate=aggregate, direct_triples=direct_triples)
                if viol[0][0] < 0:
                    num_violated += 1
                max_violations.append(viol[0])

                if direct_triples:
                    num_triples = n*(n-1)*(n-2)
                else:
                    num_triples = n*(n-1)*(n-2) // 6

                for seed in seeds:
                    num_viols = len([v for v in viols_per_seed[seed] if v < 0])
                    assert num_triples == len(viol) == len(viols_per_seed[seed]), f'{num_triples} {len(viol)} {len(viols_per_seed[seed])} {n}'
                    if num_viols > 0:
                        num_viols = int(custom_round((num_viols / num_triples) * 100))
                        num_triples_violation[seed][num_viols] = num_triples_violation[seed].get(num_viols,0) + 1

            max_violations = sorted(max_violations)
            max_viols_per_agg[aggregate] = max_violations

            violations_distr_per_seed = [[] for _ in range(100)]
            for seed in seeds:
                violations_distr = [0] * 101
                for k,v in num_triples_violation[seed].items():
                    violations_distr[k] += v
                for i in range(len(violations_distr)-2, -1, -1):
                    violations_distr[i] += violations_distr[i+1]
                violations_distr = violations_distr[1:]
                assert len(violations_distr) == len(violations_distr_per_seed), f'{len(violations_distr_per_seed)} {len(violations_distr)}'
                for i in range(len(violations_distr)):
                    violations_distr_per_seed[i].append(violations_distr[i])

            avg_std_violations_distr = []
            std_violations_distr = []
            for v in violations_distr_per_seed:
                avg_std_violations_distr.append((sum(v) / len(v), np.std(v)))
            viol_distr_per_agg[aggregate] = avg_std_violations_distr
            print(f'{num_violated}/{len(uci_ids)} datasets violated TI')

        maxviolname = 'max_viol_direct_triples' if direct_triples else 'max_viol_undirect_triples'    
        violdistrname = 'viol_distr_direct_triples' if direct_triples else 'viol_distr_undirect_triples'    
        if use_all_datasets:
            maxviolname += '_all'
            violdistrname += '_all'
        build_plot(max_viols_per_agg, extension, maxviolname, y_label='Max Additive Violation', title=f'Max Violation across Datasets', y_lim=[-1.0001, 1], figsize=(9,6))

        title = 'Distribution of Number of Violations (Directed)' if direct_triples else 'Distribution of Number of Violations (Undirected)'
        build_plot(viol_distr_per_agg, extension, violdistrname, x_label='Percentage of violations', y_label='Number of datasets with at least that many violations', title=title, y_lim=None, round_x_line=True, show_marker=False, linewidth=3)

