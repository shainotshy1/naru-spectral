import collections
import numpy as np
import datasets
import torch
import glob
import re
import made
from tqdm import tqdm

import estimators as estimators_lib
from eval_model import ReportModel, SaveEstimators, RunN, Query

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
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [table.columns[i].DistributionSize() >= 10 for i in idxs]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    return idxs, ops, vals[idxs]

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    layers = 4
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
        do_direct_io_connections=False,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=True,
        fixed_ordering=fixed_ordering,
        column_masking=False,
    )

    return model

def setup_data_model_eval(seed, table_name, target_ckpt, device):
    table = load_data(table_name)
    
    model = MakeMade(
        scale=128,
        cols_to_train=table.columns,
        seed=seed,
        fixed_ordering=None
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

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device', DEVICE)

    seed = 1234
    rng = np.random.RandomState(seed)
    num_filters = rng.choice(np.arange(3, 6))

    table_name = 'dmv-tiny'
    target_ckpt = glob.glob('./models/dmv-tiny*.pt')[0]
    table, est, oracle_est = setup_data_model_eval(seed, table_name, target_ckpt, DEVICE)


    # str_repr = ""
    # for col, op, val in zip(cols, ops, vals):
    #     str_repr += f"{col.name}{op}{val}, "
    # print(f"Query: {str_repr[:-2]}")

    # print(f"True Cardinality: {true_card}")

    num_queries = 1000
    
    for _ in tqdm(range(num_queries)):
        col_idxs, ops, vals = gen_query(table, rng, num_filters=num_filters)
        cols = np.take(table.columns, col_idxs)
        true_card = calc_cardinality(table, cols, ops, vals)

        query = (cols, ops, vals)
        Query([est],
            False,
            oracle_card=true_card,
            query=query,
            table=table,
            oracle_est=oracle_est)
    
    print(est.name, 'max', np.round(np.max(est.errs), 3), '99th',
               np.round(np.quantile(est.errs, 0.99), 3), '95th', np.round(np.quantile(est.errs, 0.95), 3),
              'median', np.round(np.quantile(est.errs, 0.5), 3))

    # print(est.est_cards, est.true_cards)

    err_csv = "results1.csv"
    SaveEstimators(err_csv, [est])
    print('...Done, result:', err_csv)

if __name__ == "__main__":
    main()

    # num_queries = 10
    # RunN(table,
    #     table.columns,
    #     [est],
    #     rng=rng,
    #     num=num_queries,
    #     log_every=1,
    #     num_filters=num_filters,
    #     oracle_cards=None,
    #     oracle_est=oracle_est)