import collections
import math
import numpy as np
import datasets
import torch
import glob
import re
import made
from tqdm import tqdm
import copy
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import estimators as estimators_lib
from eval_model import ReportModel, SaveEstimators, RunN, Query
from spectral_estimator import SpectralEstimator

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

def gen_query(table, rng, num_filters = 5):
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

    return idxs, ops, target_vals

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None, column_masking=False, residual=True, layers=4, direct_io=False):
    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        layers,
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

def setup_data_model_eval(rng, table_name, target_ckpt, device, max_rows=None):
    table = load_data(table_name)

    if table_name == 'dmv-tiny':
        model = MakeMade(
            scale=128,
            cols_to_train=table.columns,
            seed=0,
            fixed_ordering=None,
        ).to(device)
    else:
        model = MakeMade(
            scale=256,
            cols_to_train=table.columns,
            seed=0,
            fixed_ordering=None,
            column_masking=True,
            layers=5,
            direct_io=True
        ).to(device)

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

    print(f"Subsampling {max_rows} rows")
    table.EnableSubsample(max_rows, rng)

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
                                            shortcircuit=False)
    
    est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)

    oracle_est = estimators_lib.Oracle(table)

    return table, est, oracle_est

def execute_on_est(est, true_card, query, table, oracle_est):
    Query([est],
        False,
        oracle_card=true_card,
        query=query,
        table=table,
        oracle_est=oracle_est)

def train_spectral(rng, oracle_est, table, num_masks=1000, avg_n=1, max_chunks=2, p=0.2):
    oracle = copy.deepcopy(oracle_est)
    spec_est = SpectralEstimator(table, rng, max_chunks=max_chunks)
    spec_est.train(oracle, num_masks=num_masks, avg_n=avg_n, p=p)
    return spec_est

def print_est(est):
    print(est.name, 'max', np.round(np.max(est.errs), 3), '99th',
               np.round(np.quantile(est.errs, 0.99), 3), '95th', np.round(np.quantile(est.errs, 0.95), 3),
              'median', np.round(np.quantile(est.errs, 0.5), 3), 'mean', np.round(np.mean(est.errs), 3))

def plot_estimators_histograms(ests, filename="histograms.png", target_stat='err', title="", label='', alpha=0.4, colors=None):
    plt.figure(figsize=(8, 6))
    
    if colors is None:
        colors = plt.cm.tab10.colors # type: ignore

    for i, est in enumerate(ests):
        data = getattr(est, target_stat)
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 500)
        plt.fill_between(x, kde(x), alpha=alpha, color=colors[i % len(colors)], label=est.name)
        plt.plot(x, kde(x), color=colors[i % len(colors)], linewidth=1)

    plt.title(title)
    plt.xlabel(label)
    # plt.xlim(1, 100)
    plt.ylabel("Density")
    plt.legend()
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

    max_rows = 1000
    table_name = 'dmv' #'dmv'
    target_ckpt = glob.glob('./models/dmv-7.3MB*.pt')[0] #'dmv-7.3MB*'
    table, naru_est, oracle_est = setup_data_model_eval(rng, table_name, target_ckpt, DEVICE, max_rows=max_rows)
    naru_est.name = "Naru"

    print("Training Spectral")
    num_masks = 1000
    avg_n = 10
    # Masked Cardinality Estimator
    max_chunks = 3
    p = 0.05
    spec_est = train_spectral(rng, oracle_est, table, num_masks=num_masks, avg_n=avg_n, max_chunks=max_chunks, p=p)
    spec_est.name = f"MCE-1k-c{max_chunks}"

    num_filters = rng.choice(np.arange(3, 8))
    num_queries = 1000
    
    print("Evaluating...")
    for _ in tqdm(range(num_queries)):
        col_idxs, ops, vals = gen_query(table, rng, num_filters=num_filters)
        cols = np.take(table.columns, col_idxs)
        true_card = oracle_est.Query(cols, ops, vals)

        query = (cols, ops, vals)
        execute_on_est(naru_est, true_card, query, table, None)
        execute_on_est(spec_est, true_card, query, table, None)

    print_est(spec_est)
    print_est(naru_est)

    colors = ['#C877E3', '#7796E3', '#B8EB9D']
    plot_estimators_histograms([naru_est, spec_est], filename="fig_err.png", target_stat='errs', label='Estimation Error', title="Cardinality Error Distributions (DMV-Tiny)", colors=colors)
    plot_estimators_boxplots([naru_est, spec_est], filename="fig_query_dur.png", target_stat='query_dur_ms', label='Execution Duration (ms)', title="Cardinality Execution Distributions (DMV-Tiny)")

    # Cherry pick outliers (slowest MCE is faster than the fastest Naru)

    SaveEstimators("results_naru.csv", [naru_est])
    SaveEstimators("results_spec.csv", [spec_est])
    print('...Done')

if __name__ == "__main__":
    main()