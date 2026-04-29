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

from query_finder import QueryFinder, _compute_cardinalities

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

def setup_data_model_eval(rng, table_name, target_ckpt, device, max_rows=None):
    table = load_data(table_name)

    print(f"Subsampling {max_rows} rows")
    table.EnableSubsample(max_rows, rng)

    oracle_est = estimators_lib.Oracle(table)

    return table, oracle_est

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 286
    rng = np.random.RandomState(seed)
    
    num_train = 10000

    max_rows = 10000
    table_name = 'dmv'
    target_ckpt = glob.glob('./models/dmv-tiny*.pt')[0]
    table, oracle_est = setup_data_model_eval(rng, table_name, target_ckpt, DEVICE, max_rows=max_rows)
    rows = min(table.cardinality, max_rows) if max_rows is not None else table.cardinality

    num_threads = 4

    ind_est = IndepEstimator(table, table_name + f"-{rows}")

    query_finder = QueryFinder(table, oracle_est, num_val_chunks=2)

    query_finder.train(seed, ind_est, num_train, expand_n=100, num_threads=num_threads)
    benchmark_queries = query_finder.generate(rng, num_queries=100, max_spec_order=None)

    # benchmark_queries = query_finder.generate_mh(rng, ind_est, num_queries=20, num_iterations=1000)

    print("Evaluating benchmark queries...")
    true_cards = _compute_cardinalities(benchmark_queries, query_finder.baseline_estimator, num_threads=1)
    cards = _compute_cardinalities(benchmark_queries, ind_est, num_threads=1)
    for i, (q, true_card, card) in enumerate(zip(benchmark_queries, true_cards, cards)):
        card, true_card = max(card, 1), max(true_card, 1) # Avoid divide-by-zero
        print(f"Query {i+1}:")
        for c,op,v in zip(q[0], q[1], q[2]):
            print(f"     {c.name} {str(op)} {v}")
        print(f"    True Cardinality: {true_card}")
        print(f"    Indep Estimation: {card}")
        print(f"    Q-Error: {max(true_card, card) / min(true_card, card)}")
        print()

    print('...Done')

if __name__ == "__main__":
    main()