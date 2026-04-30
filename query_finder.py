from multiprocessing.pool import Pool
import os
from matplotlib.pyplot import clf, table
import numpy as np
import copy
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import lightgbm as lgb
from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver
import math

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from mce_estimator import MCE_Estimator

import matplotlib.pyplot as plt

def _compute_cardinalities_chunk(args):
    queries_chunk, oracle_est = args
    oracle = copy.deepcopy(oracle_est)
    
    results = []
    for cols, ops, vals in tqdm(queries_chunk):
        results.append(oracle.Query(cols, ops, vals))
    return results

def _compute_cardinalities(queries, oracle_est, num_threads=4):
    num_threads = min(num_threads, len(queries))
    
    n = len(queries)
    stride = (n + num_threads - 1) // num_threads
    chunks = [
        queries[i:i+stride]
        for i in range(0, n, stride)
    ]

    with Pool(num_threads) as pool:
        results = pool.map(
            _compute_cardinalities_chunk,
            [(chunk, oracle_est) for chunk in chunks]
        )

    cardinalities = np.concatenate([np.array(r) for r in results])

    return cardinalities


class QueryFinder:
    def __init__(self, table, baseline_estimator, num_val_chunks=5):
        self.table = table
        self.baseline_estimator = baseline_estimator

        self.num_val_chunks = num_val_chunks
        self.col_range_map = {}
        self.val_map = {}
        self.inv_val_map = {}
        curr_idx = -1
        for c in self.table.Columns():
            val_map = {}
            inv_map = {}
            self.val_map[c.name] = val_map
            self.inv_val_map[c.name] = inv_map

            num_chunks = min(self.num_val_chunks, len(c.all_distinct_values))
            chunk_size = (len(c.all_distinct_values) + num_chunks - 1) // num_chunks
                
            prev_curr = curr_idx + 1
            
            vals = np.array(c.all_distinct_values)
            clean = np.array([v for v in vals if not (isinstance(v, float) and math.isnan(v))])
            
            curr_chunk = -1
            for i, v in enumerate(sorted(clean)):
                targ_chunk = i // chunk_size

                if targ_chunk != curr_chunk:
                    curr_idx += 1
                    curr_chunk += 1
                    inv_map[curr_idx] = []

                val_map[v] = curr_idx
                inv_map[curr_idx].append(v)

            self.col_range_map[c.name] = (prev_curr, curr_idx + 1)

        self.encoding_length = curr_idx + 1
        print("Encoding Length:", self.encoding_length)

    def _rand_query(self, rng):
        # Do not generate queries with nan in them
        num_filters = rng.randint(1, min(15, len(self.table.columns) + 1))        
        idxs = rng.choice(len(self.table.columns), replace=False, size=num_filters)
        ops = rng.choice(['=', ">=", "<="], size=num_filters) # add 'null' filter
        
        vals = [] 
        for i in idxs:
            options = self.table.columns[i].all_distinct_values
            v = rng.choice([val for val in options if not (isinstance(val, float) and math.isnan(val))])
            vals.append(v)

        ops_all_eqs = ['='] * num_filters
        sensible_to_do_range = [self.table.columns[i].DistributionSize() >= 10 for i in idxs]
        ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

        cols = np.take(self.table.columns, idxs)
        query = (cols, ops, vals)

        return query

    def _encode(self, query):
        # encode a query as a vector of features for the estimator
        ops_allowed = ["=", "<=", ">="]
        columns, operators, vals = query
        assert all(op in ops_allowed for op in operators)

        vec = np.zeros(self.encoding_length, dtype=int)
        for c, op, v in zip(columns, operators, vals):
            start, end = self.col_range_map[c.name]
            if isinstance(v, float) and math.isnan(v):
                vec[end - 1] = 1 # Let last chunk be for NaN values
            else:
                v_idx = self.val_map[c.name][v]    
                if op == "=":
                    vec[v_idx] = 1
                elif op == "<=":
                    vec[start:end] = np.arange(start, end) <= v_idx
                elif op == ">=":
                    vec[start:end] = np.arange(start, end) >= v_idx

        return vec

    def _rand_decode(self, rng, encoding, n):
        assert len(encoding) == self.encoding_length
        queries = set()
        for _ in range(n):
            columns, operators, vals = [], [], []
            for col_name, (start, end) in self.col_range_map.items():
                col_encoding = encoding[start:end]
                if np.any(col_encoding):
                    c = self.table.GetColumn(col_name)
                    
                    idxs = np.asarray(col_encoding).nonzero()[0] + start
                    if len(idxs) == 1:
                        vals_lst = self.inv_val_map[col_name][idxs[0]]
                        op = "="
                    elif idxs[0] == start:
                        vals_lst = self.inv_val_map[col_name][idxs[-1]]
                        op = "<="
                    else:   
                        vals_lst = self.inv_val_map[col_name][idxs[0]]
                        op = ">="

                    v = rng.choice(vals_lst)
                    columns.append(c)
                    operators.append(op)
                    vals.append(v)

            queries.add((tuple(columns), tuple(operators), tuple(vals)))

        queries = list(queries)

        return queries

    def train(self, seed, targ_estimator, num_queries, expand_n=5, num_threads=4):
        rng = np.random.RandomState(seed)
        encodings = []
        queries = []
        prefix_sum = [0]
        for i in tqdm(range(num_queries), desc="Generating training queries"):
            init_query = self._rand_query(rng)
            # query_expansion = [init_query]
            encoding = self._encode(init_query)
            query_expansion = self._rand_decode(rng, encoding, n=expand_n)
            
            encodings.append(encoding)
            queries.extend(query_expansion)
            
            prefix_sum.append(prefix_sum[-1] + len(query_expansion))

        oracle_path = f'datasets/qf_{self.table.name}_cards_rows={self.table.cardinality}_n={num_queries}_chunks={self.num_val_chunks}_expansion={expand_n}_seed={seed}.npy'
        preloaded_cards = os.path.exists(oracle_path)

        if preloaded_cards:
            print("Loading precomputed cardinalities...")
            true_cards = np.load(oracle_path)
        else:
            print("Computing baseline cardinalities...")
            true_cards = _compute_cardinalities(queries, self.baseline_estimator, num_threads=num_threads)
            np.save(oracle_path, np.array(true_cards))
            print(f"Saved oracle cards to: {oracle_path}")

        print("Computing estimator cardinalities...")
        cards = _compute_cardinalities(queries, targ_estimator, num_threads=num_threads)

        print("Training estimator...")
        # Avoid divide-by-zero
        cards = np.maximum(cards, 1)
        true_cards = np.maximum(true_cards, 1)
        q_errors = np.maximum(cards, true_cards) / np.minimum(cards, true_cards)


        # ------------------------------------------------------------------------- #
        # train_prop = 0.9
        # train_n = int(train_prop * len(queries))
        # train_queries, valid_queries = queries[:train_n], queries[train_n:]
        # train_errs, valid_errs = q_errors[:train_n], q_errors[train_n:]
        # q_err_predictor = MCE_Estimator(self.table, rng, method="gbt")
        # q_err_predictor.train(train_queries, train_errs, targ_score='mae')

        # predictions = []
        # for q in valid_queries:
        #     pred = q_err_predictor.Query(q[0], q[1], q[2], store=False, card_project=False)
        #     predictions.append(pred)

        # mae_error_pred = (np.abs(np.array(predictions) - np.array(valid_errs))).mean()
        # print("Q-error Predictor MAE:", mae_error_pred)
        # # Pearson correlation (linear relationship)
        # pearson_corr, pearson_p = pearsonr(valid_errs, predictions)
        # print("Pearson correlation:", pearson_corr)
        # print("Pearson p-value:", pearson_p)

        # # Spearman correlation (rank-based, monotonic relationship)
        # spearman_corr, spearman_p = spearmanr(valid_errs, predictions)
        # print("Spearman correlation:", spearman_corr)
        # print("Spearman p-value:", spearman_p)
        # ------------------------------------------------------------------------- #

        print(f"Cardinality stats: mean={np.mean(true_cards):.4f}, median={np.median(true_cards):.4f}, 90th percentile={np.percentile(true_cards, 90):.4f}, 99th percentile={np.percentile(true_cards, 99):.4f}, max={np.max(true_cards)}")
        print(f"Q-Error stats: mean={np.mean(q_errors):.4f}, median={np.median(q_errors):.4f}, 90th percentile={np.percentile(q_errors, 90):.4f}, 99th percentile={np.percentile(q_errors, 99):.4f}, max={np.max(q_errors)}")

        fig, ax = plt.subplots(figsize=(14, 2))

        ax.scatter(q_errors, np.zeros(len(q_errors)), color="steelblue", s=6,  alpha=0.5)

        ax.set_xscale("log")
        ax.set_xlabel("Q-Error")
        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)
        plt.title(f"Independence Model Q-Error Distribution: {self.table.name}")
        plt.tight_layout()
        plt.savefig(f"err_highlights_ind_{len(q_errors)}_{self.table.name}.png", dpi=300)
        plt.close()

        avg_q_errors = []
        for i in range(num_queries):
            start, end = prefix_sum[i], prefix_sum[i+1]
            avg_q_error = np.max(q_errors[start:end])
            avg_q_errors.append(avg_q_error)

        X = np.array(encodings)
        y = np.array(avg_q_errors)


        train_prop = 0.9
        train_n = int(train_prop * len(encodings))
        train_queries, valid_queries = X[:train_n], X[train_n:]
        train_errs, valid_errs = y[:train_n], y[train_n:]

        # model = Lasso(max_iter=10000)

        # param_grid = {
        #     'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        # }

        # grid_search = GridSearchCV(
        #     model,
        #     param_grid=param_grid,
        #     cv=5,
        #     scoring='r2',
        #     verbose=0
        # )

        model = lgb.LGBMRegressor(verbose=-1, n_jobs=1)

        param_grid = {
            'num_leaves': [30, 50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, None],
        }
        
        grid_search = GridSearchCV(
            model, # type: ignore
            param_grid=param_grid,
            cv=5, 
            scoring='neg_mean_absolute_error',
            verbose=0,
            n_jobs=1
        )

        grid_search.fit(train_queries, train_errs)
        self.model, self.score = grid_search.best_estimator_, grid_search.best_score_

        print(f"Done! [MAE = {self.score}]")

        predictions = self.model.predict(valid_queries) #type: ignore

        mae_error_pred = (np.abs(np.array(predictions) - np.array(valid_errs))).mean()
        print("Q-error Predictor MAE:", mae_error_pred)
        # Pearson correlation (linear relationship)
        pearson_corr, pearson_p = pearsonr(valid_errs, predictions)
        print("Pearson correlation:", pearson_corr)
        print("Pearson p-value:", pearson_p)

        # Spearman correlation (rank-based, monotonic relationship)
        spearman_corr, spearman_p = spearmanr(valid_errs, predictions)
        print("Spearman correlation:", spearman_corr)
        print("Spearman p-value:", spearman_p)


    def generate(self, rng, num_queries, max_spec_order=None):
        print("Identifying maximizing query encoding...")
        
        # a = np.array(self.model.coef_.flatten().tolist())
        # # b = self.model.intercept_
        # best_encoding = np.zeros_like(a, dtype=int)

        # if max_spec_order is None:
        #     best_encoding[a > 0] = 1
        # else:
        #     best_indices = np.argpartition(a, -max_spec_order)[-max_spec_order:]
        #     best_encoding[best_indices] = 1
        #     best_encoding[a <= 0] = 0

        # print(best_encoding, type(best_encoding), best_encoding.shape)
        
        exact_solver = ExactSolver(maximize=True, max_solution_order=max_spec_order)

        fourier_dict = lgboost_to_fourier(self.model)

        sorted_fourier = sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        print(len(sorted_fourier))
        fourier_dict_trunc = dict(sorted_fourier[:2000]) # TODO: Make inner-range queries possible

        exact_solver.load_fourier_dictionary(fourier_dict_trunc)
        best_encoding = np.array(exact_solver.solve())
        
        print("Expected score: ", self.model.predict(best_encoding.reshape(1, -1))[0]) # type: ignore

        print("Decoding maximizing query encoding...")
        queries = self._rand_decode(rng, best_encoding, n=num_queries)
        return queries
    
    def generate_mh(self, rng, targ_estimator, num_queries, num_iterations=1000):
        print("Generating random initial query...")
        num_iterations = max(num_iterations, num_queries)

        def calc_qerror(query):
            true_card = max(_compute_cardinalities([query], self.baseline_estimator)[0], 1)
            est_card = max(_compute_cardinalities([query], targ_estimator)[0], 1)
            q_error = max(true_card, est_card) / min(true_card, est_card)
            return q_error / self.table.cardinality # Normalize by total rows to keep rewards in a reasonable range

        curr_query = self._rand_query(rng)
        all_queries = [curr_query]
        all_rewards = [calc_qerror(curr_query)]

        beta = 0.1 # Temperature parameter for acceptance probability
        
        acceptance = 0
        for _ in range(num_iterations):
            init_query = self._rand_query(rng)
            encoding = self._encode(init_query)
            prop_query = self._rand_decode(rng, encoding, n=1)[0]

            prop_q_error = calc_qerror(prop_query) # Normalize by total rows to keep rewards in a reasonable range
            curr_q_error = all_rewards[-1]
            
            log_accept_prob = (prop_q_error-curr_q_error) / beta
            accept_prob = min(1, math.exp(log_accept_prob))

            if rng.rand() < accept_prob:
                acceptance += 1
                curr_query = prop_query
                all_queries.append(curr_query)
                all_rewards.append(prop_q_error)
        
        print(f"MH Acceptance Rate: {acceptance / num_iterations:.4f}")

        # Save reward trajectory
        plt.plot(all_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Q-Error")
        plt.title(r"Metropolis-Hastings Q-Error Trajectory ($\beta=%.2f$)" % beta)
        plt.savefig(f"mh_trajectory_{self.table.name}_cards_rows={self.table.cardinality}_n={num_queries}_iter={num_iterations}.png")
        plt.clf()

        return all_queries[:num_queries]