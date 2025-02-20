import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_cosmed_time_series_data(df_full: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Extract the time series data
    df_time_series = df_full.iloc[:, 9:]

    # Step 2: Merge the first two rows into a single header row
    merged_headers = [
        f"{col[0]} ({col[1]})" if pd.notna(col[1]) else str(col[0])
        for col in zip(df_time_series.iloc[0], df_time_series.iloc[1])
    ]

    # Step 3: Assign the merged headers to the DataFrame
    df_time_series.columns = merged_headers

    # Step 4: Remove the first two rows which were used as headers
    df_time_series = df_time_series.drop(index=[0, 1]).reset_index(drop=True)

    # Step 5: Drop rows where all columns are NaN
    df_time_series.dropna(how='all', inplace=True)

    for col in df_time_series.columns:
        if col == 't (s)':
            # Step 1: Convert `datetime.time` to strings if needed
            df_time_series[col] = df_time_series[col].apply(
                lambda x: x.strftime('%H:%M:%S') if isinstance(x, datetime.time) else x
            )
            # Step 2: Convert the strings to timedeltas
            df_time_series[col] = pd.to_timedelta(df_time_series[col], errors='coerce')
        else:
            df_time_series[col] = pd.to_numeric(df_time_series[col], errors='coerce')

    return df_time_series


def extract_cosmed_meta_data(df_full: pd.DataFrame) -> dict:
    df_meta = df_full.iloc[:14, :9]
    metadata = {}
    for _, row in df_meta.iterrows():
        # Iterate over the columns in pairs (key, value)
        for col in range(0, len(row) - 1, 2):
            key = row[col]
            if pd.isna(key):
                continue
            # Ensure that col + 1 is within bounds before accessing it
            value = row[col + 1] if col + 1 < len(row) else None
            if pd.notna(value):
                metadata[key] = value
    return metadata


def read_cosmed_excel(path_file: Path) -> tuple[pd.DataFrame, dict]:
    # Read the entire Excel sheet
    df_sheet = pd.read_excel(path_file, header=None)

    # Extract time series data and metadata
    df = extract_cosmed_time_series_data(df_sheet)
    meta = extract_cosmed_meta_data(df_sheet)

    return df, meta


def plot_param(df: pd.DataFrame, param: str, smooth: bool = False):
    fig, ax = plt.subplots(figsize=(10, 10))
    # convert dtype: timedelta64[ns] to seconds for plotting
    x = df['t (s)'].dt.total_seconds()
    if smooth:
        ax.plot(x, df[param].rolling(window=20, center=True).mean(), label=f'Smoothed {param}', color='b', linewidth=2)
        ax.plot(x, df[param], label=param, color='b', linewidth=0.1)
    else:
        ax.plot(x, df[param], label=param, color='b', linewidth=2)
    ax.set_ylabel(param)
    ax.grid()
    ax.set_title(param)
    ax.set_xlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    path = Path(
        r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\AFT_Habituation_2\Durability\data\spiro")
    subjects = [s for s in path.iterdir()]
    for sub in subjects:
        s_path = Path.joinpath(path, sub)
        conditions = [c for c in s_path.iterdir()]

        fig, ax = plt.subplots(figsize=(16, 10))
        param = 'VO2/Kg (mL/min/Kg)'

        for con in conditions:
            if 'Non' in con.name:
                color = 'r'
            else:
                color = 'b'
            c_path = Path.joinpath(s_path, con)
            print(f"Subject = {sub.name}, Condition = {con.name}")
            files = [f for f in c_path.iterdir() if f.suffix == '.xlsx']
            if len(files) != 1:
                raise ValueError(f"Expected 1 Excel file in {c_path}, found {len(files)}")
            file = files[0]
            df, meta = read_cosmed_excel(Path.joinpath(path, file))
            x = df['t (s)'].dt.total_seconds()
            # ax.plot(x, df[param], label=param, color=color, linewidth=2)
            ax.plot(x, df[param].rolling(window=20, center=True).mean(), label=f'Smoothed {param}', color=color, linewidth=2)
            ax.plot(x, df[param], label=param, color=color, linewidth=0.1)
        ax.set_ylabel(param)
        ax.grid()
        ax.set_xlabel('Time (s)')
        path_plot = s_path.joinpath(f'overlay.svg')
        plt.savefig(path_plot)
        # plt.show()

        # plot_param(df, 'VO2/Kg (mL/min/Kg)', smooth=False)
        # plot_param(df, 'Af (1/min)', smooth=True)
        # plot_param(df, 'VE (L/min)', smooth=True)
        # plot_param(df, 'HF (bpm)', smooth=True)
        # plot_param(df, 'VT (L(btps))', smooth=True)
