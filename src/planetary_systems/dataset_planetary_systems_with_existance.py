from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


_NUMERIC_COLS = ["a", "total_mass", "r"]
_ALL_COLS = ["a", "total_mass", "r", "exists", "does_not_exist"]


class FrameStandardScaler:
    """
    Wraps sklearn StandardScaler but operates on pandas DataFrames,
    transforming only the numeric columns and leaving one-hot columns
    (exists/does_not_exist) unchanged.
    """
    def __init__(self, numeric_cols: List[str]):
        self.numeric_cols = list(numeric_cols)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "FrameStandardScaler":
        self.scaler.fit(df[self.numeric_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Accept extra columns and only inverse-transform numeric ones
        out = df.copy()
        has_all = all(c in out.columns for c in self.numeric_cols)
        if has_all:
            out[self.numeric_cols] = self.scaler.inverse_transform(out[self.numeric_cols])
        return out


@dataclass
class _SystemExample:
    values: np.ndarray  # shape (max_len, 5)
    length: int         # number of real planets


class _PlanetaryDataset(Dataset):
    def __init__(self, tensors: List[np.ndarray], lengths: List[int]):
        self.tensors = tensors
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.tensors[idx], dtype=torch.float32)  # (T, 5)
        n = int(self.lengths[idx])
        return x, n


class DatasetPlanetarySystemsWithExistance:
    """
    Loads planetary systems from CSV, normalizes numeric columns, adds
    exists/does_not_exist one-hot, and pads to max planets.

    Accepts either:
      - Long format: columns include ['system_id', 'planet_idx', 'a', 'total_mass', 'r']
      - Wide format: columns like 'a_1','a_2',... and similarly for 'total_mass','r'
    """
    def __init__(self, csv_path, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0):
        self.csv_path = str(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.normalizer = FrameStandardScaler(_NUMERIC_COLS)

        df = pd.read_csv(self.csv_path)
        systems = self._parse_systems(df)  # list of dicts with keys 'a','total_mass','r' as lists

        self.num_planets = max(len(s["a"]) for s in systems) if systems else 0

        # Fit scaler on real (existing) planets only
        if systems:
            stacked = np.concatenate([
                np.column_stack([np.asarray(s["a"]), np.asarray(s["total_mass"]), np.asarray(s["r"])])
                for s in systems if len(s["a"]) > 0
            ], axis=0)
            df_fit = pd.DataFrame(stacked, columns=_NUMERIC_COLS)
            self.normalizer.fit(df_fit)

        # Build padded tensors and DataFrame views
        examples: List[_SystemExample] = []
        norm_rows: List[Dict[str, Any]] = []
        pad_rows: List[Dict[str, Any]] = []

        for s in systems:
            n = len(s["a"])
            arr = np.zeros((self.num_planets, len(_ALL_COLS)), dtype=np.float32)

            # Normalize numeric values per planet, then insert and set exists one-hot
            if n > 0:
                raw_df = pd.DataFrame({
                    "a": s["a"],
                    "total_mass": s["total_mass"],
                    "r": s["r"],
                })
                norm_df = self.normalizer.transform(raw_df)
                for t in range(n):
                    arr[t, 0:3] = norm_df.iloc[t][_NUMERIC_COLS].to_numpy(dtype=np.float32)
                    arr[t, 3] = 1.0  # exists
                    arr[t, 4] = 0.0  # does_not_exist
                    # for normalized DF (real planets only)
                    norm_rows.append({
                        "a": arr[t, 0], "total_mass": arr[t, 1], "r": arr[t, 2],
                        "exists": 1.0, "does_not_exist": 0.0
                    })

            # Fill padding rows' one-hot
            if n < self.num_planets:
                arr[n:, 3] = 0.0
                arr[n:, 4] = 1.0

            # for padded DF (all time steps including padding)
            for t in range(self.num_planets):
                pad_rows.append({
                    "a": arr[t, 0], "total_mass": arr[t, 1], "r": arr[t, 2],
                    "exists": arr[t, 3], "does_not_exist": arr[t, 4]
                })

            examples.append(_SystemExample(values=arr, length=n))

        self._tensors = [ex.values for ex in examples]
        self._lengths = [ex.length for ex in examples]

        # For your histograms
        self.normalized = pd.DataFrame(norm_rows, columns=_ALL_COLS) if norm_rows else pd.DataFrame(columns=_ALL_COLS)
        self.padded = pd.DataFrame(pad_rows, columns=_ALL_COLS) if pad_rows else pd.DataFrame(columns=_ALL_COLS)

        # Torch dataset + dataloader
        self.dataset = _PlanetaryDataset(self._tensors, self._lengths)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def _parse_systems(self, df: pd.DataFrame) -> List[Dict[str, List[float]]]:
        # Try long format first
        if {"system_id", "planet_idx"}.issubset(df.columns) and set(_NUMERIC_COLS).issubset(df.columns):
            systems: List[Dict[str, List[float]]] = []
            for _, g in df.sort_values(["system_id", "planet_idx"]).groupby("system_id"):
                systems.append({
                    "a": g["a"].astype(float).tolist(),
                    "total_mass": g["total_mass"].astype(float).tolist(),
                    "r": g["r"].astype(float).tolist(),
                })
            return systems

        # Otherwise try wide format with suffixed columns like a_1, a_2, ...
        col_map: Dict[str, List[Tuple[int, str]]] = {}
        for base in _NUMERIC_COLS:
            pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
            matches: List[Tuple[int, str]] = []
            for c in df.columns:
                m = pat.match(c)
                if m:
                    matches.append((int(m.group(1)), c))
            matches.sort(key=lambda x: x[0])
            col_map[base] = matches

        if any(len(col_map[base]) > 0 for base in _NUMERIC_COLS):
            systems: List[Dict[str, List[float]]] = []
            for _, row in df.iterrows():
                sys_dict = {base: [] for base in _NUMERIC_COLS}
                # Assume all bases share the same set of indices; use intersection just in case
                indices_sets = [set(i for i, _c in col_map[base]) for base in _NUMERIC_COLS if col_map[base]]
                common_indices = sorted(set.intersection(*indices_sets)) if indices_sets else []
                for idx in common_indices:
                    vals = []
                    ok = True
                    for base in _NUMERIC_COLS:
                        col_name = f"{base}_{idx}"
                        if col_name in df.columns:
                            vals.append(float(row[col_name]))
                        else:
                            ok = False
                            break
                    if ok:
                        sys_dict["a"].append(vals[0])
                        sys_dict["total_mass"].append(vals[1])
                        sys_dict["r"].append(vals[2])
                systems.append(sys_dict)
            return systems

        # As a last resort, if the CSV is already one system per row with plain columns (no indexing),
        # treat each row as a 1-planet system.
        if set(_NUMERIC_COLS).issubset(df.columns):
            systems = [{
                "a": [float(row["a"])],
                "total_mass": [float(row["total_mass"])],
                "r": [float(row["r"])],
            } for _, row in df.iterrows()]
            return systems

        raise ValueError(
            "CSV format not recognized. Expected either long format "
            "(system_id, planet_idx, a, total_mass, r) or wide format "
            "(a_1, a_2, ..., total_mass_1, ..., r_1, ...)."
        )
