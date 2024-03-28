from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_divergence_curves(divergence_curves: pd.DataFrame, path_data_out: Path, location_string: str):
    df = divergence_curves
    average = df.stack().mean(axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.plot(average, 'k', linewidth=3)
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='x', nbins=20)
    ax.tick_params(direction='in')
    ax.set_ylabel('avg(log(divergence))', fontsize=18)
    fig.set_tight_layout(True)
    fig.suptitle(f"Divergence Curves ({location_string.title().replace('_', ' ')})", fontsize=20)
    fig.savefig(path_data_out.joinpath(f'divergence_curves_{location_string}.svg'))
