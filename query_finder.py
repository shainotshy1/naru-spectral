from multiprocessing.pool import Pool

from matplotlib.pyplot import clf
import numpy as np
import copy
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import lightgbm as lgb
from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver


class QueryFinder:
    def __init__(self, table, baseline_estimator, num_val_chunks=5):
        self.table = table
        self.baseline_estimator = baseline_estimator

        self.col_range_map = {}
        self.val_map = {}
        self.inv_val_map = {}
        curr_idx = -1
        for c in self.table.Columns():
            val_map = {}
            inv_map = {}
            self.val_map[c.name] = val_map
            self.inv_val_map[c.name] = inv_map

            num_chunks = min(num_val_chunks, len(c.all_distinct_values))
            chunk_size = (len(c.all_distinct_values) + num_chunks - 1) // num_chunks
                
            prev_curr = curr_idx + 1
            
            curr_chunk = -1
            for i, v in enumerate(sorted(c.all_distinct_values)):
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

    def _compute_cardinalities_chunk(self, args):
        queries_chunk, oracle_est = args
        oracle = copy.deepcopy(oracle_est)
        
        results = []
        for cols, ops, vals in tqdm(queries_chunk):
            results.append(oracle.Query(cols, ops, vals))
        return results
    
    def _compute_cardinalities(self, queries, oracle_est, num_threads=4):
        num_threads = min(num_threads, len(queries))
        
        n = len(queries)
        stride = (n + num_threads - 1) // num_threads
        chunks = [
            queries[i:i+stride]
            for i in range(0, n, stride)
        ]

        with Pool(num_threads) as pool:
            results = pool.map(
                self._compute_cardinalities_chunk,
                [(chunk, oracle_est) for chunk in chunks]
            )

        cardinalities = np.concatenate([np.array(r) for r in results])

        return cardinalities

    def _encode(self, query):
        # encode a query as a vector of features for the estimator
        ops_allowed = ["=", "<=", ">="]
        columns, operators, vals = query
        assert all(op in ops_allowed for op in operators)

        vec = np.zeros(self.encoding_length, dtype=int)
        for c, op, v in zip(columns, operators, vals):
            start, end = self.col_range_map[c.name]
            v_idx = self.val_map[c.name][v]
            if op == "=":
                vec[v_idx] = 1
            elif op == "<=":
                vec[start:end] = np.arange(start, end) <= v_idx
            elif op == ">=":
                vec[start:end] = np.arange(start, end) >= v_idx

        return vec

    def _rand_decode(self, encoding, n):
        assert len(encoding) == self.encoding_length
        queries = []
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

                    v = np.random.choice(vals_lst)
                    columns.append(c)
                    operators.append(op)
                    vals.append(v)

            queries.append((columns, operators, vals))

        return queries

    def train(self, targ_estimator, init_queries, expand_n=5):
        encodings = []
        queries = []
        prefix_sum = [0]
        for i, query in tqdm(enumerate(init_queries), desc="Expanding training queries"):
            encoding = self._encode(query)
            
            query_expansion = self._rand_decode(encoding, n=expand_n)
            encodings.append(encoding)
            queries.extend(query_expansion)
            prefix_sum.append(prefix_sum[-1] + len(query_expansion))

        print("Computing baseline cardinalities...")
        true_cards = self._compute_cardinalities(queries, self.baseline_estimator)

        print("Computing estimator cardinalities...")
        cards = self._compute_cardinalities(queries, targ_estimator)

        print("Training estimator...")
        # Avoid divide-by-zero
        cards = np.maximum(cards, 1)
        true_cards = np.maximum(true_cards, 1)
        q_errors = np.maximum(cards, true_cards) / np.minimum(cards, true_cards)

        avg_q_errors = []
        for i in range(len(init_queries)):
            start, end = prefix_sum[i], prefix_sum[i+1]
            avg_q_error = np.mean(q_errors[start:end])
            avg_q_errors.append(avg_q_error)

        X = np.array(encodings)
        y = np.array(avg_q_errors)

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
            scoring='r2',
            verbose=0,
            n_jobs=1
        )


        # self.model = Lasso(max_iter=10000)

        # param_grid = {
        #     'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        # }

        # grid_search = GridSearchCV(
        #     self.model,
        #     param_grid=param_grid,
        #     cv=5,
        #     scoring='r2',
        #     verbose=0
        # )
            
        grid_search.fit(X, y)
        self.model, self.score = grid_search.best_estimator_, grid_search.best_score_

        print(f"Done! [R^2 = {self.score}]")

    def generate(self, num_queries, max_spec_order=None):
        print("Identifying maximizing query encoding...")
        exact_solver = ExactSolver(maximize=True, max_solution_order=max_spec_order)

        fourier_dict = lgboost_to_fourier(self.model)

        sorted_fourier = sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        fourier_dict_trunc = dict(sorted_fourier[:1000])

        exact_solver.load_fourier_dictionary(fourier_dict_trunc)
        best_encoding = np.array(exact_solver.solve())

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
        # print("Expected score: ", self.model.predict(best_encoding.reshape(1, -1))[0])
        
        print("Decoding maximizing query encoding...")
        queries = self._rand_decode(best_encoding, n=num_queries)
        return queries