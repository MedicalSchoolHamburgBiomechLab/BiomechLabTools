import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from sys import platform

import pandas as pd

gettrace = getattr(sys, 'gettrace', None)

if gettrace():
    DEBUG = True
else:
    DEBUG = False


def get_cpu_core_count(adj: int = -1):
    if DEBUG:
        return 1
    if adj > 0:
        adj = 0
        warnings.warn('cpu count to use can only be adjusted by negative values')
    cnt = os.cpu_count()
    if 'win' in platform and ('darwin' not in platform):
        return int(cnt / 2) + adj
    return cnt + adj


def process_on_dataframe(df: pd.DataFrame, func, multiprocess: bool = True, *args, **kwargs):
    df_out = pd.DataFrame(columns=df.columns, index=df.index, dtype=object)
    if multiprocess:
        n_cores = get_cpu_core_count(adj=0)
        with ProcessPoolExecutor(n_cores) as executor:
            for col in df.columns:
                for ind, row in df.iterrows():
                    signal = df.at[ind, col]
                    if type(signal) == float:
                        continue
                    df_out.at[ind, col] = executor.submit(func, signal, *args, **kwargs)
            for ind, row in df_out.iterrows():
                for k, v in df_out.loc[ind].items():
                    if type(v) == float:
                        continue
                    try:
                        df_out.at[ind, k] = v.result()
                    except Exception as e:
                        print(f'Error in {ind} {k}')
                        print(e)
                        df_out.at[ind, k] = None
    else:
        for col in df.columns:
            for ind, row in df.iterrows():
                signal = df.at[ind, col]
                if type(signal) == float:
                    continue
                df_out.at[ind, col] = func(signal, *args, **kwargs)
    return df_out
