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

class MCE_Estimator(CardEst):
    def __init__(self, table, rng, linear=False):
        super(MCE_Estimator, self).__init__()
        self.name = 'MDCE' if not linear else 'Lasso'
        self.model = None
        self.cv_r2 = None
        self.table = table
        self.rng = rng
        self.linear = linear

        # Count distinct of strings - how? Use sentiment score to approximate the value so it can be passed to the GBT
        # IDEA: How to do cardinality estimation with string arguments? Use sentiment analysis as input rather than the string themselves
        # Limitations: GBTs are powerful but this work is limited by its assumption of the data being floats. How can we deal with string data?

        print("Setting up structures...", end="", flush=True)

        ops_allowed = ['=', '>=', '<=']
        self.op_code_len = int(np.ceil(np.log2(len(ops_allowed))))
        self.op_map = {op : tuple([int(bit) for bit in format(i, '08b')][:self.op_code_len]) for i, op in enumerate(ops_allowed)}
        self.inv_op_map = {tuple([int(bit) for bit in format(i, '08b')][:self.op_code_len]) : op for i, op in enumerate(ops_allowed)}

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

        print("Done!")

    def _query_to_vec(self, columns, operators, vals):
        assert all([op in self.op_map or op == 'null' for op in operators])
        n = len(self.col_map)
        v_len = n + n * self.op_code_len + n # selected columns + opcodes per columns + predicate value
        vec = np.zeros(v_len)
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
        vals = [self.inv_val_map[cols[i]][vec[v_offset + i]] for i in col_indexes]
        
        return cols, ops, vals

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        assert self.model is not None, "Must train model first!"
        mask = self._query_to_vec(columns, operators, vals)

        self.OnStart()
        c = self.model.predict(mask.reshape(1, -1)) # type: ignore
        self.OnEnd()

        return np.maximum(np.round(c[0]), 1) # type: ignore

    def train(self, queries, cardinalities):
        n = len(self.col_map)
        query_vecs = []
        for (cols, ops, vals) in queries:
            vec = self._query_to_vec(cols, ops, vals)
            query_vecs.append(vec)

        query_vecs = np.array(query_vecs)
        vals = np.array(cardinalities)
    
        if self.linear:
            model = Lasso(max_iter=10000)

            # Define hyperparameter grid for Lasso
            param_grid = {
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            }

            grid_search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=5,
                scoring='r2',
                verbose=0
            )
        else:
            model = lgb.LGBMRegressor(verbose=-1, n_jobs=4)

            # Define the hyperparameter grid
            param_grid = {
                'num_leaves': [31,50],
                'learning_rate': [0.01, 0.1],
                'max_depth': [4,6]
            }
            
            # param_grid = {
            #     'max_depth': [3, 5, None],
            #     'n_estimators': [500, 1000, 5000],
            #     'learning_rate': [0.01, 0.1]
            # }

            # Perform GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid=param_grid, # type: ignore
                cv=5, scoring='r2', verbose=0
            )
            # model = RandomForestRegressor(random_state=42, n_jobs=1)

            # # Simpler hyperparameter grid
            # param_grid = {
            #     'n_estimators': [100, 200],
            #     'max_depth': [4, 6],
            #     'min_samples_split': [2, 5]
            # }

            # grid_search = GridSearchCV(
            #     model,
            #     param_grid=param_grid,
            #     cv=5,
            #     scoring='r2',
            #     verbose=0,
            #     njobs=4
            # )
            # self.linear = True

        grid_search.fit(query_vecs, vals)
        self.model, self.cv_r2 = grid_search.best_estimator_, grid_search.best_score_

        print(f"R2: {self.cv_r2}")

        return self.cv_r2
    
    def save_model(self, path):
        assert self.model is not None, 'Must train model first!'
        if self.linear:
            with open(path,'wb') as f:
                pickle.dump(self.model,f)
        else:
            self.model.booster_.save_model(path) # type: ignore

    def load_model(self, path):
        if self.linear:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = lgb.Booster(model_file=path)