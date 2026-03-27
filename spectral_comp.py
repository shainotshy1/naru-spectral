import collections
import numpy as np
import datasets
import torch
import glob
import re
import made
from tqdm import tqdm
import copy

import estimators as estimators_lib
from eval_model import ReportModel, SaveEstimators, RunN, Query
from spectral_estimator import SpectralEstimator

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

def calc_cardinality(table, columns, operators, values):
    mask = np.ones(len(table.data), dtype=bool)
    for col, op, val in zip(columns, operators, values):
        col_data = table.data[col.name] 
        if op == '=':
            target_op = '=='
        else:
            target_op = op
        mask &= eval("col_data {} val".format(target_op))

    true_cardinality = mask.sum()
    return true_cardinality

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
    
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    ops = rng.choice(['='], size=num_filters) # , ">=", "<="
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [table.columns[i].DistributionSize() >= 10 for i in idxs]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    return idxs, ops, vals[idxs]

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

def setup_data_model_eval(seed, table_name, target_ckpt, device, max_rows=None):
    table = load_data(table_name)

    if table_name == 'dmv-tiny':
        model = MakeMade(
            scale=128,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=None,
        ).to(device)
    else:
        model = MakeMade(
            scale=256,
            cols_to_train=table.columns,
            seed=seed,
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
    table.EnableSubsample(max_rows)

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

def train_spectral(oracle_est, table, num_masks=1000, avg_n=1):
    oracle = copy.deepcopy(oracle_est)
    spec_est = SpectralEstimator(table)
    spec_est.train(oracle, num_masks=num_masks, avg_n=avg_n)
    return spec_est

def print_est(est):
    print(est.name, 'max', np.round(np.max(est.errs), 3), '99th',
               np.round(np.quantile(est.errs, 0.99), 3), '95th', np.round(np.quantile(est.errs, 0.95), 3),
              'median', np.round(np.quantile(est.errs, 0.5), 3), 'mean', np.round(np.mean(est.errs), 3))

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 1234
    rng = np.random.RandomState(seed)

    max_rows = None
    table_name = 'dmv-tiny' #'dmv'
    target_ckpt = glob.glob('./models/dmv-tiny*.pt')[0] #'dmv-7.3MB*'
    table, naru_est, oracle_est = setup_data_model_eval(seed, table_name, target_ckpt, DEVICE, max_rows=max_rows)

    print("Training Spectral")
    num_masks = 1000
    avg_n = 5
    spec_est = train_spectral(oracle_est, table, num_masks=num_masks, avg_n=avg_n)

    num_filters = rng.choice(np.arange(3, 8))
    num_queries = 100
    
    print("Evaluating...")
    for _ in tqdm(range(num_queries)):
        col_idxs, ops, vals = gen_query(table, rng, num_filters=num_filters)
        cols = np.take(table.columns, col_idxs)
        true_card = calc_cardinality(table, cols, ops, vals)

        query = (cols, ops, vals)
        execute_on_est(naru_est, true_card, query, table, oracle_est)
        execute_on_est(spec_est, true_card, query, table, oracle_est)

    print_est(spec_est)
    print_est(naru_est)

    SaveEstimators("results_naru.csv", [naru_est])
    SaveEstimators("results_spec.csv", [spec_est])
    print('...Done')

if __name__ == "__main__":
    main()