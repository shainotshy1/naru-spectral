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
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

import estimators as estimators_lib
from eval_model import ReportModel, SaveEstimators, RunN, Query
from mce_estimator import MCE_Estimator

from ind_estimator import IndepEstimator

import time
from multiprocessing import Pool

from query_finder import QueryFinder


import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

def load_data(ds_name):
    assert ds_name in ['dmv-tiny', 'dmv']
    if ds_name == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
        table.name = 'dmv-tiny'
    elif ds_name == 'dmv':
        table = datasets.LoadDmv()
        table.name = 'dmv'
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
        ops = rng.choice(['=', ">=", "<="], size=num_filters) # add 'null' filter
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

    oracle_est = estimators_lib.Oracle(table)

    return table, oracle_est, est

def execute_on_est(est, true_card, query, table, oracle_est):
    Query([est],
        False,
        oracle_card=true_card,
        query=query,
        table=table,
        oracle_est=oracle_est)

def train_mce(rng, queries, cardinalities, table, path='mce_model.txt', method="gbt"):
    spec_est = MCE_Estimator(table, rng, method=method)
    print("Training model...")
    spec_est.train(queries, cardinalities)
    spec_est.save_model(path)
    print(f"Saved model to: {path}")
    return spec_est

def print_est(est, attribute='errs'):
    print('max', np.round(np.max(getattr(est, attribute)), 3), '99th',
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

def compute_cardinalities_chunk(args):
    queries_chunk, oracle_est = args
    oracle = copy.deepcopy(oracle_est)
    
    results = []
    for cols, ops, vals in tqdm(queries_chunk):
        results.append(oracle.Query(cols, ops, vals))
    return results

def get_train_valid_data(rng, table, table_name, oracle_est, max_rows, seed, num_train, num_valid, recollect_data=False, num_threads=4):
    total_num = num_train + num_valid

    oracle_path = f'datasets/{table_name}_cards_rows={max_rows}_n={total_num}_seed={seed}.npy'
    preloaded_cards = not recollect_data and os.path.exists(oracle_path)
    
    queries = []
    for _ in tqdm(range(total_num)):
        num_filters = rng.choice(np.arange(3, 8))#rng.choice(np.arange(1, len(table.columns) + 1))
        col_idxs, ops, vals = gen_query(table, rng, num_filters=num_filters, nan_check=False)
        cols = np.take(table.columns, col_idxs)
        query = (cols, ops, vals)

        queries.append(query)

    n = len(queries)

    if preloaded_cards:
        print(f"Loading oracle cards from: {oracle_path}")
        cardinalities = np.load(oracle_path)
    else:
        print("Calculating Cardinalities...")
        start = time.time()

        stride = (n + num_threads - 1) // num_threads
        chunks = [
            queries[i:i+stride]
            for i in range(0, n, stride)
        ]

        with Pool(num_threads) as pool:
            results = pool.map(
                compute_cardinalities_chunk,
                [(chunk, oracle_est) for chunk in chunks]
            )

        cardinalities = np.concatenate([np.array(r) for r in results])

        print(f"All processes complete! [{time.time() - start} secs.]")

        np.save(oracle_path, np.array(cardinalities))
        print(f"Saved oracle cards to: {oracle_path}")

    train = (queries[:num_train], cardinalities[:num_train])
    valid = (queries[num_train:], cardinalities[num_train:])

    return train, valid

def gen_card_model(method, retrain_model, table, table_name, rows, seed, training_size, rng, test_data):
    path = f'models/mce_{table_name}_rows={rows}_seed={seed}_method={method}_samples={training_size}'
    path += '.pkl' if method == "gbt" else '.txt'
    if not retrain_model and os.path.exists(path):
        spec_est = MCE_Estimator(table, rng, method=method)
        spec_est.load_model(path)
        print(f"Loaded model from: {path}")
    else:
        test_q, test_c = test_data
        spec_est = train_mce(rng, test_q, test_c, table, path=path, method=method)

    return spec_est

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 286
    rng = np.random.RandomState(seed)
    
    recollect_data = False
    retrain_model = False

    num_train = 5000
    num_valid = 0

    target_algs = ["ind"] #, "linear"] # , "forest", "gbt", "linear"
    test_ests = []
    get_naru = "naru" in target_algs

    max_rows = None
    table_name = 'dmv-tiny'
    target_ckpt = glob.glob('./models/dmv-tiny*.pt')[0]
    table, oracle_est, naru_est = setup_data_model_eval(rng, table_name, target_ckpt, DEVICE, max_rows=max_rows, get_naru=get_naru)
    rows = min(table.cardinality, max_rows) if max_rows is not None else table.cardinality

    if get_naru:
        test_ests.append(naru_est)

    num_threads = 16
    # train_data, valid_data = get_train_valid_data(rng, table, table_name, oracle_est, rows, seed, num_train, num_valid, recollect_data=recollect_data, num_threads=num_threads)
    
    # for method in target_algs:
    #     if method != "naru" and method != "ind":
    #         print(f"Generating {method.upper()} Estimator...")
    #         est = gen_card_model(method, retrain_model, table, table_name, rows, seed, num_train, rng, test_data)
    #         test_ests.append(est)
    #     if method == "ind":
    #         print(f"Generating {method.upper()} Estimator...")
    #         est = IndepEstimator(table, table_name + f"-{rows}")
    #         test_ests.append(est)

    ind_est = IndepEstimator(table, table_name + f"-{rows}")

    query_finder = QueryFinder(table, oracle_est, num_val_chunks=2)

    # ex_query = train_data[0][0]
    # ex_encoding = query_finder._encode(ex_query)
    # ex_decoding = query_finder._rand_decode(ex_encoding, n=1)
    # print("Original:")
    # for c,op,v in zip(ex_query[0], ex_query[1], ex_query[2]):
    #     print(f"    {c.name} {str(op)} {v}")
    # print("Decoded:")
    # for c,op,v in zip(ex_decoding[0][0], ex_decoding[0][1], ex_decoding[0][2]):
    #     print(f"    {c.name} {str(op)} {v}")

    # print("Example query encoding:", ex_encoding)


    query_finder.train(seed, ind_est, num_train, expand_n=5)
    benchmark_queries = query_finder.generate(rng, num_queries=5, max_spec_order=None)

    print("Evaluating benchmark queries...")
    true_cards = query_finder._compute_cardinalities(benchmark_queries, query_finder.baseline_estimator, num_threads=1)
    cards = query_finder._compute_cardinalities(benchmark_queries, ind_est, num_threads=1)
    for i, (q, true_card, card) in enumerate(zip(benchmark_queries, true_cards, cards)):
        card, true_card = max(card, 1), max(true_card, 1) # Avoid divide-by-zero
        print(f"Query {i+1}:")
        for c,op,v in zip(q[0], q[1], q[2]):
            print(f"     {c.name} {str(op)} {v}")
        print(f"    True Cardinality: {true_card}")
        print(f"    Indep Estimation: {card}")
        print(f"    Q-Error: {max(true_card, card) / min(true_card, card)}")
        print()


    # print("Evaluating...")
    # valid_q, valid_c = valid_data
    # for i in tqdm(range(num_valid)):
    #     query, true_card = valid_q[i], valid_c[i]
    #     for est in test_ests:
    #         execute_on_est(est, true_card, query, table, None)
    #         if est.name == "Naru" and est.errs[-1] > 8:
    #             est.errs.pop()
    #             est.est_cards.pop()
    #             est.true_cards.pop()
    #             break # skip rest since invalid query - hardcoding since there is some encoding issue
    #         # if est.name == "Naru":
    #         #     print(true_card, est.est_cards[-1], est.errs[-1])

    # if "gbt" in target_algs:
    #     spec_est = test_ests[target_algs.index("gbt")]
    #     print("Example encoding:")
    #     ex_idx = 0
    #     cols, ops, vals = valid_q[ex_idx]
    #     true_card = valid_c[ex_idx]
    #     print("    Query:")
    #     for c,op,v in zip(cols, ops, vals):
    #         print(f"     {c.name} {str(op)} {v}")

    #     ex_vec = spec_est._query_to_vec(cols, ops, vals,)
    #     print(f"    Encoding: {ex_vec}")
    #     inv_cols, inv_ops, inv_vals = spec_est._vec_to_query(ex_vec)
    #     print("    Inverse: ")
    #     for c,op,v in zip(inv_cols, inv_ops, inv_vals):
    #         print(f"     {c.name} {str(op)} {v}")
            
    #     print(f"    Cardinality: {true_card}")
    #     print(f"    Prediction: {spec_est.Query(cols, ops, vals, store=False)}")

    # for est in test_ests:  
    #     print(f"---{est.name}---")
    #     print_est(est, attribute='errs')
    #     print_est(est, attribute='query_dur_ms')
    #     print()

    # test_ests[0], test_ests[1] = test_ests[1], test_ests[0] # Put independent in the front

    # -------------------------- #
    # for i in range(1, len(test_ests)):
    #     print(test_ests[i].name)
    #     errs1 = np.array(test_ests[0].errs)
    #     errs2 = np.array(test_ests[i].errs)

    #     top = 10
    #     res = []
    #     for top in np.arange(0, 105, 5):
    #         top_worst1 = np.where(errs1 >= np.percentile(errs1, 100 - top))[0]
    #         top_worst2 = np.where(errs2 >= np.percentile(errs2, 100 - top))[0]
    #         shared = np.setdiff1d(top_worst1, np.setxor1d(top_worst1, top_worst2))
    #         prop_shared = len(shared) / len(top_worst1)
    #         res.append(prop_shared)
    #     print(res)

    #     from scipy.stats import pearsonr, spearmanr

    #     pearson_corr, _ = pearsonr(errs1, errs2)
    #     spearman_corr, _ = spearmanr(errs1, errs2)

    #     print(f"Full estimator Pearson correlation: {pearson_corr:.4f}")
    #     print(f"Full estimator Spearman correlation: {spearman_corr:.4f}")

    #     percent = 10#10 / num_valid * 100
    #     worst_idx = np.where(errs1 >= np.percentile(errs1, 100 - percent))[0]
    #     normal_idx = np.setdiff1d(np.arange(len(errs2)), worst_idx)

    #     fig, ax = plt.subplots(figsize=(14, 2))

    #     ax.scatter(errs2[normal_idx], np.zeros(len(normal_idx)), color="steelblue", s=6,  alpha=0.5, label="est2")
    #     ax.scatter(errs2[worst_idx],  np.zeros(len(worst_idx)),  color="crimson",   s=12, alpha=0.9, label=f"est2 @ est1 worst {percent}%")

    #     ax.set_xscale("log")
    #     ax.set_xlabel("Q-Error")
    #     ax.yaxis.set_visible(False)
    #     ax.spines[["left", "top", "right"]].set_visible(False)
    #     ax.legend(fontsize=9)
    #     plt.tight_layout()
    #     plt.savefig(f"err_highlights_{test_ests[i].name}.png", dpi=300)
    #     plt.close()
    # -------------------------- #

    # colors = ['#C877E3', '#7796E3', '#B8EB9D']
    # plot_estimators_boxplots(test_ests, filename="fig_err.png", target_stat='errs', label='Estimation Error', title="")
    # plot_estimators_boxplots(test_ests, filename="fig_query_dur.png", target_stat='query_dur_ms', label='Execution Duration (ms)', title="")

    # SaveEstimators("results.csv", test_ests)
    print('...Done')

if __name__ == "__main__":
    main()