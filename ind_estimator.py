from estimators import CardEst, OPS

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json

from math import isnan


# ---------------------------------------------------------------------------
# JSON encode / decode helpers (unchanged)
# ---------------------------------------------------------------------------

def _encode_value(val):
    if isinstance(val, pd.Timestamp):
        return {"__type__": "timestamp", "v": val.isoformat()}
    if isinstance(val, np.datetime64):
        return {"__type__": "datetime64", "v": pd.Timestamp(val).isoformat()}
    if isinstance(val, float) and isnan(val):
        return {"__type__": "nan"}
    if isinstance(val, np.integer):
        return {"__type__": "int", "v": int(val)}
    if isinstance(val, np.floating):
        return {"__type__": "float", "v": float(val)}
    return {"__type__": "raw", "v": val}


def _decode_value(obj):
    t = obj["__type__"]
    if t == "timestamp":
        return pd.Timestamp(obj["v"])
    if t == "datetime64":
        return np.datetime64(pd.Timestamp(obj["v"]))
    if t == "nan":
        return float("nan")
    if t == "int":
        return int(obj["v"])
    if t == "float":
        return float(obj["v"])
    return obj["v"]


def _encode_counts(counts: dict) -> list:
    return [[_encode_value(v), cnt] for v, cnt in counts.items()]


def _decode_counts(pairs: list) -> dict:
    return {_decode_value(enc): cnt for enc, cnt in pairs}

# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class IndepEstimator(CardEst):
    """Uses per-column value frequencies + independence assumption."""

    def __init__(self, table, table_name, cache_dir="cache"):
        super(IndepEstimator, self).__init__()
        self.table = table
        self.table_name = table_name
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        self.col_indexes = {}
        for c in table.Columns():
            self.col_indexes[c] = len(self.col_indexes)

        self.total_rows = len(table.Columns()[0].data)

        cache_path = os.path.join(
            self.cache_dir, f"{self.table_name}_value_counts.json"
        )

        if os.path.exists(cache_path):
            self.value_counts = self._load_cache(cache_path)
            print(f"Loaded value counts from cache: {cache_path}")
        else:
            self.value_counts = self._compute_counts()
            self._save_cache(cache_path)
            print(f"Saved value counts to cache: {cache_path}")

        self.value_count_vecs = {}
        for c, counts in self.value_counts.items():
            self.value_count_vecs[c] = np.array(list(counts.keys())), np.array(list(counts.values()))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _compute_counts(self) -> dict:
        value_counts = {}
        for c in tqdm(self.table.Columns(), desc="Computing value counts"):
            counts: dict = {}
            for val in c.data:
                counts[val] = counts.get(val, 0) + 1
            value_counts[c.name] = counts
        return value_counts

    def _save_cache(self, path: str) -> None:
        serialisable = {
            col_name: _encode_counts(counts)
            for col_name, counts in self.value_counts.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f)

    def _load_cache(self, path: str) -> dict:
        with open(path, "r") as f:
            raw = json.load(f)
        return {
            col_name: _decode_counts(pairs)
            for col_name, pairs in raw.items()
        }

    # ------------------------------------------------------------------

    def __str__(self):
        return "indep_estimator"

    def _canonical_type(self, val):
        if isinstance(val, (np.integer, int)):
            return int
        if isinstance(val, (np.floating, float)):
            return float
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp
        return type(val)

    def _selectivity(self, column, operator, value):
        """Estimate selectivity for a single predicate — fully vectorized."""
        if operator not in OPS:
            raise NotImplementedError(f"Operator {operator} not supported")

        if type(value) is np.datetime64:
            value = pd.Timestamp(value)

        vals, cnts = self.value_count_vecs[column.name]

        if isinstance(value, float) and isnan(value):
            try:
                nan_mask = np.array([isinstance(v, float) and isnan(v) for v in vals])
            except Exception:
                nan_mask = np.zeros(len(vals), dtype=bool)
            matched = cnts[nan_mask].sum()        
        else:
            mask = OPS[operator](vals, value)
            matched = cnts[mask].sum()

        return int(matched) / self.total_rows

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        sel = 1.0
        for c, o, v in zip(columns, operators, vals):
            sel *= self._selectivity(c, o, v)

        estimate = int(sel * self.total_rows)

        self.OnEnd()
        return estimate