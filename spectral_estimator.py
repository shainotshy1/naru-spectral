from estimators import CardEst, OPS
import numpy as np
from tqdm import tqdm
import math
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

class SpectralEstimator(CardEst):
    def __init__(self, table, rng, max_chunks=2):
        super(SpectralEstimator, self).__init__()
        self.name = 'SpectralEst'
        self.model = None
        self.cv_r2 = None
        self.table = table
        self.rng = rng

        column_domains = [col.all_distinct_values for col in self.table.Columns()]

        self.col_chunk_map = {}
        self.col_val_map = {}
        self.inv_col_val_map = {}
        for c,d in zip(self.table.Columns(), column_domains):
            if type(d[0]) is float and math.isnan(d[0]):
                target_d = d[1:]
            else:
                target_d = d
                
            sorted_d = np.sort(target_d)
            chunks = np.array_split(sorted_d, max_chunks)
            # remove empty chunks if len(d) < max_chunks
            # also reverse order so that first (where 'nan' is placed) is the smallest length chunk
            chunks = chunks[:len(sorted_d)][::-1]
            
            print(len(d), c.distribution_size)
            self.col_chunk_map[c] = {}
            for i in range(len(chunks)):
                idx = len(self.col_val_map)
                self.col_val_map[(c, i)] = idx
                self.inv_col_val_map[idx] = (c, chunks[i])
                for v in chunks[i]:
                    self.col_chunk_map[c][v] = i
            # ALWAYS add 'nan' in first chunk
            self.col_chunk_map[c][float('nan')] = 0

    def _query_to_bitmap(self, columns, operators, vals):
        assert all([op == '=' for op in operators]) # Only allow equijoins
        n = len(self.col_val_map)
        bitmap = np.zeros(n)
        for c,v in zip(columns, vals):
            if type(v) is float and math.isnan(v):
                i = 0 # ALWAYS 'nan' in first chunk
            else:
                i = self.col_chunk_map[c][v]
            bitmap[self.col_val_map[(c, i)]] = 1
        return bitmap
    
    def _bitmap_to_query(self, bitmap):
        # Only allow equijoins
        comps = [self.inv_col_val_map[i] for i,b in enumerate(bitmap) if b == 1]
        if len(comps) == 0:
            return [], [], []
        columns, domain_chunks = zip(*comps)
        vals = [self.rng.choice(d) for d in domain_chunks]
        operators = ["="] * len(comps)
        return columns, operators, vals

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        assert self.model is not None, "Must train model first!"
        mask = self._query_to_bitmap(columns, operators, vals)

        self.OnStart()
        c = self.model.predict(mask.reshape(1, -1)) # type: ignore
        self.OnEnd()

        return np.maximum(np.round(c[0]), 1)

    def train(self, oracle_est, num_masks=1000, avg_n=1, p=0.2):
        n = len(self.col_val_map)
        all_masks = self.rng.choice(2, size=(num_masks, n), p = np.array([1-p, p])) # p probability of a 1
        values = []
        for mask in tqdm(all_masks):
            value = 0
            for _ in range(avg_n):
                cols, ops, vals = self._bitmap_to_query(mask)
                c = oracle_est.Query(cols, ops, vals)
                value += (1/avg_n) * c
            values.append(value)
        values = np.array(values)

        model = lgb.LGBMRegressor(verbose=-1, n_jobs=4)

        # Define the hyperparameter grid
        param_grid = {
            'num_leaves': [31,50],
            'learning_rate': [0.01, 0.1],
            'max_depth': [4,6]
        }

        # Perform GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid=param_grid, # type: ignore
            cv=5, scoring='r2', verbose=0
        )

        grid_search.fit(all_masks, values)
        best_model, cv_r2 = grid_search.best_estimator_, grid_search.best_score_

        self.model = best_model
        self.cv_r2 = cv_r2

        print(f"Spectal Training R2: {cv_r2}")

        return cv_r2
    
    def save_model(self, path):
        assert self.model is not None
        self.model.booster_.save_model(path) # type: ignore

    def load_model(self, path):
        self.model = lgb.Booster(model_file=path)