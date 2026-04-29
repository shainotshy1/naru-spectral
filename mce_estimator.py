from matplotlib.pyplot import clf
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from estimators import CardEst, OPS
import numpy as np
from tqdm import tqdm
import math
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pickle

from sklearn.metrics import make_scorer, mean_absolute_error

class MCE_Estimator(CardEst):
    def __init__(self, table, rng, method="gbt"):
        assert method in ["gbt", "linear", "forest"]
        super(MCE_Estimator, self).__init__()

        self.method = method
        self.name = "MCE-" + self.method.upper()
        self.model = None
        self.score = None
        self.table = table
        self.rng = rng

        print("Setting up structures...", end="", flush=True)

        ops_allowed = ['=', '>=', '<=']
        self.op_code_len = int(np.ceil(np.log2(len(ops_allowed))))
        self.op_map = {op : tuple([int(bit) for bit in format(i, '08b')][-self.op_code_len:]) for i, op in enumerate(ops_allowed)}
        self.inv_op_map = {tuple([int(bit) for bit in format(i, '08b')][-self.op_code_len:]) : op for i, op in enumerate(ops_allowed)}

        self.col_map = {}
        self.inv_col_map = {}
        for c in self.table.Columns():
            idx = len(self.col_map)
            self.col_map[c] = idx
            self.inv_col_map[idx] = c

        self.val_map = {}
        self.inv_val_map = {}
        for c in self.table.Columns():
            val_map = {}
            inv_map = {}
            self.val_map[c] = val_map
            self.inv_val_map[c] = inv_map
            for v in c.all_distinct_values:
                idx = len(val_map)
                val_map[v] = idx
                inv_map[idx] = v

        n = len(self.col_map)
        print(f"Done! Total vector lengths: [{n + n * self.op_code_len + n}]")

    def _query_to_vec(self, columns, operators, vals):
        assert all([op in self.op_map or op == 'null' for op in operators])
        n = len(self.col_map)
        v_len = n + n * self.op_code_len + n # selected columns + opcodes per columns + predicate value
        vec = np.zeros(v_len, dtype=int)
        for c, op, v in zip(columns, operators, vals):
            col_idx = self.col_map[c]
            vec[col_idx] = 1
            if op != 'null':
                op_idx = n + self.op_code_len * col_idx
                val_idx = n * (1 + self.op_code_len) + col_idx
                vec[op_idx : op_idx + self.op_code_len] = self.op_map[op]
                vec[val_idx] = self.val_map[c][v]
        return vec
    
    def _vec_to_query(self, vec):
        n = len(self.col_map)
        col_indexes = [i for i in range(n) if vec[i] == 1]
        if len(col_indexes) == 0:
            return [], [], []
        cols = [self.inv_col_map[i] for i in col_indexes]
        ops = [self.inv_op_map[tuple(vec[n + self.op_code_len * i:n + self.op_code_len * (i+1)])] for i in col_indexes]
        v_offset = n * (1 + self.op_code_len)
        vals = [self.inv_val_map[c][int(vec[v_offset + i])] for i, c in zip(col_indexes, cols)]
        
        return cols, ops, vals

    def Query(self, columns, operators, vals, store=True, card_project=True):
        assert len(columns) == len(operators) == len(vals)
        assert self.model is not None, "Must train model first!"
        mask = self._query_to_vec(columns, operators, vals)

        if store: self.OnStart()
        c = self.model.predict(mask.reshape(1, -1)) # type: ignore
        if store: self.OnEnd()

        if card_project:
            return int(np.maximum(np.round(c[0]), 0)) # type: ignore
        else: 
            return c[0] # type: ignore

    def _q_error(self, y_true, y_pred):
        eps = 1e-10
        y_true = np.maximum(y_true, eps)
        y_pred = np.maximum(y_pred, eps)

        q = np.maximum(y_pred / y_true, y_true / y_pred)
        return np.median(q)

    def train(self, queries, cardinalities, targ_score='q_error'):
        n = len(self.col_map)
        query_vecs = []
        for (cols, ops, vals) in queries:
            vec = self._query_to_vec(cols, ops, vals)
            query_vecs.append(vec)

        query_vecs = np.array(query_vecs, dtype=int)
        vals = np.array(cardinalities, dtype=int)

        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'q_error': make_scorer(self._q_error, greater_is_better=False)
        }

        assert targ_score in scoring
    
        if self.method == "linear":
            model = Lasso(max_iter=10000)

            param_grid = {
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            }

            grid_search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=5,
                scoring=scoring,
                refit=targ_score,
                verbose=0
            )
        elif self.method == "gbt":
            model = lgb.LGBMRegressor(verbose=-1, n_jobs=4)

            param_grid = {
                'num_leaves': [30, 50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, None],
                'lambda_l1': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            }
            
            grid_search = GridSearchCV(
                model, # type: ignore
                param_grid=param_grid,
                cv=5, 
                scoring=scoring, 
                refit=targ_score,
                verbose=0
            )
        elif self.method == "forest":
            model = RandomForestRegressor(verbose=0, n_jobs=4)

            param_grid = {
                'n_estimators': [10, 20, 30],
                'max_depth': [3, 5, None]
            }

            grid_search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=5,
                scoring=scoring,
                refit=targ_score
            )
        else:
            raise ValueError("Unsupported method type: ", self.method)
            
        grid_search.fit(query_vecs, vals)
        self.model, self.score = grid_search.best_estimator_, grid_search.best_score_

        print(f"{targ_score}: {self.score}")

        return self.score
    
    def save_model(self, path):
        assert self.model is not None, 'Must train model first!'
        if self.method == "gbt":
            self.model.booster_.save_model(path) # type: ignore
        else:
            with open(path,'wb') as f:
                pickle.dump(self.model,f)

    def load_model(self, path):
        if self.method == "gbt":
            self.model = lgb.Booster(model_file=path)
        else:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
