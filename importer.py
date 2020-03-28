import numpy as np
import os
import pandas as pd

def import_csv(path):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + path
    return pd.read_csv(dir_path)

def split_df(df, split=0.8):
    pt = np.random.rand(len(df)) < split
    return df[pt], df[~pt]
