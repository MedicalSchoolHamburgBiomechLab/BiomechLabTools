import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

gettrace = getattr(sys, 'gettrace', None)
DEBUG = True if gettrace() else False


def get_cpu_core_count(adjust: int = -1):
    if DEBUG:
        return 1
    if adjust > 0:
        warnings.warn('cpu count to use can only be adjusted by negative values')
        adjust = 0
    cnt = os.cpu_count() or 1  # Default to 1 if None
    if sys.platform.startswith('win'):
        return max(int(cnt / 2) + adjust, 1)
    return max(cnt + adjust, 1)


def process_on_dataframe(df: pd.DataFrame, func, multiprocess: bool = True, *args, **kwargs) -> pd.DataFrame:
    """
    Apply a function to each element in a DataFrame
    :param df: DataFrame
    :param func: function to apply
    :param multiprocess: use multiprocessing
    :param args: additional arguments for the function
    :param kwargs: additional keyword arguments for the function
    """
    if multiprocess:
        n_cores = get_cpu_core_count(adjust=0)
        with ProcessPoolExecutor(n_cores) as executor:
            # Flatten the DataFrame to a Series for easy mapping
            df_flat = df.stack(future_stack=True)
            # Submit tasks
            futures = {executor.submit(func, val, *args, **kwargs): idx for idx, val in df_flat.items()}
            # Collect results
            results = {}
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f'Error processing {idx}: {e}')
                    results[idx] = None
            # Reconstruct the DataFrame
            df_out = pd.Series(results).unstack()
    else:
        # Define a safe function that catches exceptions
        def safe_func(x):
            if pd.notnull(x):
                try:
                    return func(x, *args, **kwargs)
                except Exception as e:
                    print(f'Error processing value {x}: {e}')
                    return None
            else:
                return x

        # Flatten the DataFrame to a Series
        df_flat = df.stack(future_stack=True)
        # Apply the function
        df_flat = df_flat.map(safe_func)
        # Reconstruct the DataFrame
        df_out = df_flat.unstack()
    return df_out


def limit_to_range(x, min_n, max_n):
    if min_n > max_n:
        raise ValueError('min_n must be less than or equal to max_n')
    if x < min_n:
        return min_n
    if x > max_n:
        return max_n
    return x
