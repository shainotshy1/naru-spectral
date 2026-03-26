from estimators import CardEst, OPS
import numpy as np
from tqdm import tqdm

from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver # type:ignore

class SpectralEstimator(CardEst):
    def __init__(self, table):
        super(SpectralEstimator, self).__init__()
        self.name = 'SpectralEst'
        self.model = None
        self.cv_r2 = None
        self.table = table

        column_domains = [col.all_distinct_values for col in self.table.Columns()]

        self.col_val_map = {}
        self.inv_col_val_map = {}
        for c,d in zip(self.table.Columns(), column_domains):
            # for v in d:
            idx = len(self.col_val_map)
            self.col_val_map[c] = idx
            self.inv_col_val_map[idx] = (c, d)
                            
    def _query_to_bitmap(self, columns, operators, vals):
        assert all([op == '=' for op in operators]) # Only allow equijoins
        n = len(self.col_val_map)
        bitmap = np.zeros(n)
        for c,v in zip(columns, vals):
            bitmap[self.col_val_map[c]] = 1
        return bitmap
    
    def _bitmap_to_query(self, bitmap):
        # Only allow equijoins
        comps = [self.inv_col_val_map[i] for i,b in enumerate(bitmap) if b == 1]
        if len(comps) == 0:
            return [], [], []
        columns, domains = zip(*comps)
        vals = [np.random.choice(d) for d in domains]
        operators = ["="] * len(comps)
        return columns, operators, vals

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        assert self.model is not None, "Must train model first!"
        mask = self._query_to_bitmap(columns, operators, vals)

        self.OnStart()
        c = self.model.predict(mask.reshape(1, -1)) # type: ignore
        self.OnEnd()

        return np.maximum(np.round(c[0]), 0)

    def train(self, gen_query, oracle_est, num_masks=1000, p=0.2):
        n = len(self.col_val_map)
        all_masks = np.random.choice(2, size=(num_masks, n), p = np.array([1-p, p])) # p probability of a 1
        values = []
        for mask in tqdm(all_masks):
            cols, ops, vals = self._bitmap_to_query(mask)
            values.append(oracle_est.Query(cols, ops, vals))
        values = np.array(values)
        best_model, cv_r2 = lgboost_fit(all_masks, values)
        self.model = best_model
        self.cv_r2 = cv_r2

        print(f"Spectal Training R2: {cv_r2}")

        return cv_r2
    
    # def train(self, gen_query, oracle_est, num_masks=1000, seed=0, filter_range=None):
    #     rng = np.random.RandomState(seed)

    #     queries = []
    #     values = []
    #     all_masks = []
    #     for _ in tqdm(range(num_masks)):
    #         if filter_range is None: 
    #             num_filters = rng.choice(np.arange(3, 8))
    #         else:
    #             num_filters = rng.choice(filter_range)

    #         col_idxs, ops, vals = gen_query(self.table, rng, num_filters=num_filters)
    #         cols = [self.table.Columns()[i] for i in col_idxs]
    #         queries.append((cols, ops, vals))
    #         all_masks.append(self._query_to_bitmap(cols, ops, vals))
    #         values.append(oracle_est.Query(cols, ops, vals))
        
    #     all_masks = np.array(all_masks)
    #     values = np.array(values)
    #     best_model, cv_r2 = lgboost_fit(all_masks, values)
    #     self.model = best_model
    #     self.cv_r2 = cv_r2

    #     print(f"Spectal Training R2: {cv_r2}")

    #     return cv_r2