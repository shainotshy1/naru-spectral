import collections
import math
import numpy as np
import datasets
import torch
import glob
import re
import os
import made
from tqdm import tqdm
import copy
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import estimators as estimators_lib
from eval_model import ReportModel, SaveEstimators, RunN, Query
from mce_estimator import MCE_Estimator

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

def load_data(ds_name):
    assert ds_name in ['dmv-tiny', 'dmv']
    if ds_name == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif ds_name == 'dmv':
        table = datasets.LoadDmv()
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    return table

def gen_query(table, rng, num_filters = 5, nan_check=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values
    vals[6] = vals[6].to_datetime64()
    idxs = []
    ops = []
    target_vals = [float('nan')]
    while any([type(v) is float and math.isnan(v) for v in target_vals]):
        idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
        ops = rng.choice(['='], size=num_filters) # , ">=", "<="
        ops_all_eqs = ['='] * num_filters
        sensible_to_do_range = [table.columns[i].DistributionSize() >= 10 for i in idxs]
        ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
        target_vals = vals[idxs]
        if not nan_check:
            break

    return idxs, ops, target_vals

def MakeMade(scale, cols_to_train, seed=1234, fixed_ordering=None, column_masking=False, residual=True, layers=4, direct_io=False):
    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        layers if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding='binary',
        output_encoding='one_hot',
        embed_size=32,
        seed=seed,
        do_direct_io_connections=direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=residual,
        fixed_ordering=fixed_ordering,
        column_masking=column_masking,
    )

    return model

def setup_data_model_eval(rng, table_name, target_ckpt, device, max_rows=None, get_naru=False):
    table = load_data(table_name)

    if get_naru:
        if table_name == 'dmv-tiny':
            column_masking = False
            model = MakeMade(
                scale=128,
                cols_to_train=table.columns,
                fixed_ordering=None,
                column_masking=column_masking
            ).to(device)
        elif target_ckpt.split('/')[-1].startswith('dmv-7.3MB'):
            column_masking = True
            model = MakeMade(
                scale=256,
                cols_to_train=table.columns,
                fixed_ordering=None,
                column_masking=column_masking,
                layers=5,
                direct_io=True
            ).to(device)
        elif target_ckpt.split('/')[-1].startswith('dmv-19.8MB'):
            column_masking = True
            model = MakeMade(
                scale=128,
                cols_to_train=table.columns,
                fixed_ordering=None,
                column_masking=column_masking,
                layers=0,
                direct_io=True,
                residual=False
            ).to(device)
        else:
            raise ValueError(f"Unsupported checkpoint: {target_ckpt}")
        
        print('Loading ckpt:', target_ckpt)
        model.load_state_dict(torch.load(target_ckpt))
        model.eval()

        z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',target_ckpt)
        print(target_ckpt)
        assert z
        model_bits = float(z.group(1))
        data_bits = float(z.group(2))
        seed = int(z.group(3))
        bits_gap = model_bits - data_bits

        Ckpt = collections.namedtuple(
            'Ckpt', 'epoch model_bits bits_gap path loaded_model seed')

        ckpt = Ckpt(path=target_ckpt,
                    epoch=None,
                    model_bits=model_bits,
                    bits_gap=bits_gap,
                    loaded_model=model,
                    seed=seed)
        
        psample = 2000

        est = estimators_lib.ProgressiveSampling(ckpt.loaded_model,
                                                table,
                                                psample,
                                                device=device,
                                                shortcircuit=column_masking)
        
        # est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)
        est.name = "Naru" 
    else:
        est = None

    print(f"Subsampling {max_rows} rows")
    table.EnableSubsample(max_rows, rng)

    oracle_est = estimators_lib.Oracle(table, count_distinct=True)

    return table, est, oracle_est

def execute_on_est(est, true_card, query, table, oracle_est):
    Query([est],
        False,
        oracle_card=true_card,
        query=query,
        table=table,
        oracle_est=oracle_est)

def train_mce(rng, oracle_est, table, num_masks=1000, avg_n=1, max_chunks=2, p=0.2, path='mce_model.txt', linear=False):
    oracle = copy.deepcopy(oracle_est)
    spec_est = MCE_Estimator(table, rng, max_chunks=max_chunks, linear=linear)
    if os.path.exists(path):
        spec_est.load_model(path)
        print(f"Loaded model from: {path}")
    else:
        print("Training model...")
        spec_est.train(oracle, num_masks=num_masks, avg_n=avg_n, p=p)
        spec_est.save_model(path)
        print(f"Saved model to: {path}")
    return spec_est

def print_est(est, attribute='errs'):
    print(est.name, 'max', np.round(np.max(getattr(est, attribute)), 3), '99th',
               np.round(np.quantile(getattr(est, attribute), 0.99), 3), '95th', np.round(np.quantile(getattr(est, attribute), 0.95), 3),
              'median', np.round(np.quantile(getattr(est, attribute), 0.5), 3), 'mean', np.round(np.mean(getattr(est, attribute)), 3))

def plot_estimators_histograms(ests, filename="histograms.png", target_stat='err', title="", label='', alpha=0.4, colors=None):
    plt.figure(figsize=(8, 6))
    
    if colors is None:
        colors = plt.cm.tab10.colors # type: ignore

    for i, est in enumerate(ests):
        data = getattr(est, target_stat)
        kde = gaussian_kde(data, bw_method=0.05)
        x = np.linspace(min(data), max(data), 500)
        plt.fill_between(x, kde(x), alpha=alpha, color=colors[i % len(colors)], label=est.name)
        plt.plot(x, kde(x), color=colors[i % len(colors)], linewidth=1)

    plt.title(title)
    plt.xlabel(label)
    # plt.xlim(1, 25)
    plt.ylabel("Density")
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_estimators_boxplots(ests, filename="boxplots.png", target_stat='err', title="", label='', outliers=False):
    data = [getattr(est, target_stat) for est in ests]
    labels = [est.name for est in ests]
    plt.figure(figsize=(6, 6))
    plt.boxplot(data, tick_labels=labels, showfliers=outliers)
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 286
    rng = np.random.RandomState(seed)

    max_rows = 1000000
    table_name = 'dmv'
    target_ckpt = glob.glob('./models/dmv-tiny*.pt')[0]
    table, naru_est, oracle_est = setup_data_model_eval(rng, table_name, target_ckpt, DEVICE, max_rows=max_rows, get_naru=False)
    rows = min(table.cardinality, max_rows) if max_rows is not None else table.cardinality

    linear = False
    # num_masks = 80
    avg_n = 1
    max_chunks = 1
    p = 0.5
    # while num_masks < 400:
    for num_masks in [115]:
        print("Number of masks:", num_masks)
        path=f'models/mce_{table_name}_chunks={max_chunks}_rows={max_rows}_masks={num_masks}_p={p}_seed={seed}_linear={linear}'
        path += '.pkl' if linear else '.txt'
        spec_est = train_mce(rng, oracle_est, table, num_masks=num_masks, avg_n=avg_n, max_chunks=max_chunks, p=p, path=path, linear=linear)
        spec_est.name = f"MDCE-{num_masks}-c{max_chunks}"
        num_masks = round(num_masks * 1.2)

        num_queries = 1000

        oracle_path = f'datasets/{table_name}_cards_rows_{rows}_seed_{seed}_queries_{num_queries}.npy'
        preloaded_cards = os.path.exists(oracle_path)
        if preloaded_cards:
            print(f"Loading oracle cards from: {oracle_path}")
            oracle_cards = np.load(oracle_path)
        else:
            oracle_cards = []

        rng = np.random.RandomState(seed + 1) # Reset range to ensure same queries whether or not we train MCE or load from file
        print("Evaluating...")
        for i in tqdm(range(num_queries)):
            num_filters = rng.choice(np.arange(1, len(table.columns) + 1))
            col_idxs, ops, vals = gen_query(table, rng, num_filters=num_filters, nan_check=False)
            cols = np.take(table.columns, col_idxs)

            if preloaded_cards:
                true_card = oracle_cards[i]
            else:
                true_card = oracle_est.Query(cols, ops, vals)
                oracle_cards.append(true_card)

            query = (cols, ops, vals)
            execute_on_est(spec_est, true_card, query, table, None)

        if not preloaded_cards:
            np.save(oracle_path, np.array(oracle_cards))
            print(f"Saved oracle cards to: {oracle_path}")

        print(f"[{num_masks}]", end=' ')
        print_est(spec_est, attribute='errs')
        print_est(spec_est, attribute='query_dur_ms')

    colors = ['#C877E3', '#7796E3', '#B8EB9D']
    plot_estimators_histograms([spec_est], filename="fig_err.png", target_stat='errs', label='Estimation Error', title="", colors=colors)
    plot_estimators_boxplots([spec_est], filename="fig_query_dur.png", target_stat='query_dur_ms', label='Execution Duration (ms)', title="")

    SaveEstimators("results_spec.csv", [spec_est])
    print('...Done')

if __name__ == "__main__":
    main()