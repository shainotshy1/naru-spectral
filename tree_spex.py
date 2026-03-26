from gurobipy import Model, GRB, LinExpr, Env
from itertools import chain, combinations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import logging

def lgboost_fit(X, y):
    model = lgb.LGBMRegressor(verbose=-1, n_jobs=4)

    # Define the hyperparameter grid
    param_grid = {
        'num_leaves': [31,50],
        'learning_rate': [0.01, 0.1],
        'max_depth': [4,6]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid=param_grid,
        cv=5, scoring='r2', verbose=0
    )

    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_score_


def lgboost_tree_to_fourier(tree_info):
    """
    Strips the Fourier coefficients from an LGBoost tree
    Code adapted from:
        Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
        "Amortized SHAP values via sparse Fourier function approximation."
        arXiv preprint arXiv:2410.06300 (2024).
    """

    def fourier_tree_sum(left_fourier, right_fourier, feature):
        final_fourier = {}
        all_freqs_tuples = set(left_fourier.keys()).union(right_fourier.keys())
        for freq_tuple in all_freqs_tuples:
            final_fourier[freq_tuple] = (left_fourier.get(freq_tuple, 0) + right_fourier.get(freq_tuple, 0)) / 2
            current_freq_set = set(freq_tuple)
            feature_set = {feature}
            united_set = current_freq_set.union(feature_set)
            final_fourier[tuple(sorted(united_set))] = (0.5 * left_fourier.get(freq_tuple, 0)
                                                        - 0.5 * right_fourier.get(freq_tuple, 0))
        return final_fourier

    def dfs(node):
        if 'leaf_value' in node:  # Leaf node in LightGBM JSON
            return {tuple(): node['leaf_value']}
        else:  # Split node
            left_fourier = dfs(node['left_child'])
            right_fourier = dfs(node['right_child'])
            feature_index = node['split_feature']  # Feature index for LightGBM
            return fourier_tree_sum(left_fourier, right_fourier, feature_index)

    return dfs(tree_info['tree_structure'])


def lgboost_to_fourier(model):
    final_fourier = []
    dumped_model = model.booster_.dump_model()
    for tree_info in dumped_model['tree_info']:
        final_fourier.append(lgboost_tree_to_fourier(tree_info))

    combined_fourier = {}
    for fourier in final_fourier:
        for k, v in fourier.items():
            tuple_k = [0] * model.n_features_
            for feature in k:
                tuple_k[feature] = 1
            tuple_k = tuple(tuple_k)
            if tuple_k in combined_fourier:
                combined_fourier[tuple_k] += v
            else:
                combined_fourier[tuple_k] = v
    return combined_fourier


class ExactSolver:
    def __init__(self, maximize=True, max_solution_order=None):
        logging.getLogger('gurobipy').setLevel(logging.WARNING)
        self.maximize = maximize
        self.max_solution_order = max_solution_order
        self.loaded = False
        self.key = {
            #"WLSACCESSID": "38d41f1f-f181-4d1d-9d6b-63f1dfff392c",
            #"WLSSECRET": "8ed03468-18d7-4e2a-adef-a8d6703c8d2a",
            #"LICENSEID": 2403918,
            "LICENSEID": 2680322,
        }

        with suppress_stdout():
            self.env = Env(params=self.key)

    def fourier_to_mobius(self, fourier_dict):
        """
        Convert Fourier coefficients to Mobius coefficients.

        Parameters:
        - fourier_dict: A dictionary of Fourier coefficients.

        Returns:
        - A dictionary of Mobius coefficients.
        """
        if len(fourier_dict) == 0:
            return {}
        else:
            unscaled_mobius_dict = {}
            for loc, coef in fourier_dict.items():
                real_coef = np.real(coef)
                for subset in self.all_subsets(np.nonzero(loc)[0]):
                    one_hot_subset = tuple([1 if i in subset else 0 for i in range(self.n)])
                    if one_hot_subset in unscaled_mobius_dict:
                        unscaled_mobius_dict[one_hot_subset] += real_coef
                    else:
                        unscaled_mobius_dict[one_hot_subset] = real_coef

            # multiply each entry by (-2)^(cardinality)
            return {loc: val * np.power(-2.0, np.sum(loc)) for loc, val in unscaled_mobius_dict.items()}

    def load_fourier_dictionary(self, fourier_dictionary):
        self.loaded = True
        self.fouier_dictionary = fourier_dictionary
        assert len(self.fouier_dictionary) > 0, "Empty Dictionary"
        self.n = len(list(self.fouier_dictionary.keys())[0])
        self.mobius_dictionary = self.fourier_to_mobius(self.fouier_dictionary)
        self.baseline_value = self.mobius_dictionary[tuple([0] * self.n)] if tuple(
            [0] * self.n) in self.mobius_dictionary else 0
        
        self.initialize_model()

    def all_subsets(self, iterable, order=None):
        """
        Returns all subset tuples of the given iterable.
        """
        if not order:
            return list(chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))
        else:
            return list(chain.from_iterable(combinations(iterable, r) for r in range(order, order + 1)))

    def initialize_model(self):
        self.model = Model("Mobius Maximization Problem", env=self.env)
        vars = [(tuple(np.nonzero(key)[0]), val) for key, val in self.mobius_dictionary.items() if sum(key) > 0]
        self.locs, self.coefs = [i[0] for i in vars], [i[1] for i in vars]
        locs_set = set(self.locs)
        # print(f"Number of locations: {len(self.locs)}")

        # Define the variables and objective function
        y = self.model.addVars(len(self.locs), vtype=GRB.BINARY, name="y")
        self.model.setObjective(
            sum(self.coefs[i] * y[i] for i in range(len(self.locs))),
            GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        )

        # Constraint 1: y_S <= y_R \forall R \subset S, \forall S
        count_constraint_1, count_constraint_2 = 0, 0
        for i, loc in enumerate(self.locs):
            for loc_subset in self.all_subsets(loc, order=len(loc) - 1):
                if loc_subset in locs_set and loc_subset != loc:
                    j = self.locs.index(loc_subset)
                    self.model.addConstr(y[i] <= y[j])
                    count_constraint_1 += 1

        # Constraint 2: \sum_{i \in S} y_{i} <= |S| + y_S - 1, \forall S
        for i, loc in enumerate(self.locs):
            if len(loc) > 1:
                expr = LinExpr()
                for idx in loc:
                    if (idx,) in locs_set:
                        expr.add(y[self.locs.index((idx,))])
                self.model.addConstr(expr <= len(loc) + y[i] - 1)
                count_constraint_2 += 1
        # print(f"Constraint 1: {count_constraint_1}")
        # print(f"Constraint 2: {count_constraint_2}")

        # (Optional) Constraint 3: \sum_{i \in n} y_{i} <= n
        if self.max_solution_order is not None:
            expr = LinExpr()
            for idx in range(self.n):
                if (idx,) in locs_set:
                    expr.add(y[self.locs.index((idx,))])
            self.model.addConstr(expr <= self.max_solution_order)

    def solve(self):
        assert self.loaded, "ERORR: must load fourier dictionary first"
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        # Print the optimal values
        argmax = [0] * self.n
        for i, var in enumerate(self.model.getVars()):
            if len(self.locs[i]) == 1 and var.x > 0.5:
                argmax[self.locs[i][0]] = 1

        # print(f"Est. {'argmax' if self.maximize else 'argmin'} {argmax} with est value {np.round(self.model.objVal + self.baseline_value, 3)}")
        return argmax