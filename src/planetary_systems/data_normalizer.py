import pandas as pd
import numpy as np 


class Normalizer:
    def __init__(self):
        self.columns_to_log = ["a", "total_mass", "r"]
        self.columns_to_normalize = ["a", "total_mass", "r"]
        self.medians = None
        self.stds = None
        
    def fit(self, df: pd.DataFrame):
        logged = df.copy()
        for col in self.columns_to_log:
            logged[col] = np.log(logged[col])
        self.medians = logged[self.columns_to_normalize].median()
        self.stds = logged[self.columns_to_normalize].std()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns_to_log:
            df[col] = np.log(df[col])
        df[self.columns_to_normalize] = (
            df[self.columns_to_normalize] - self.medians
        ) / self.stds
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns_to_normalize] = (
            df[self.columns_to_normalize] * self.stds + self.medians
        )
        for col in self.columns_to_log:
            df[col] = np.exp(df[col])
        return df
