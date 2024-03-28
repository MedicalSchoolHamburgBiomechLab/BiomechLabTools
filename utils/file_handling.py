from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    try:
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                if item.shape:
                    ans[key] = item[:]
                else:
                    # ans[key] = np.array([0])
                    ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    except Exception as e:
        print(e)
    return ans


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            h5file[path + key] = []
            # raise ValueError('Cannot save %s type'%type(item))


def get_folder_list(path: Path, absolute: bool = False, sort: bool = False):
    folders_list = [f for f in path.iterdir() if f.is_dir()]
    if sort:
        folders_list.sort()
    if absolute:
        return folders_list
    return [f.name for f in folders_list]


def get_participant_folder_list(path: Path, sort: bool = True, absolute: bool = False, id_prefix: str = None):
    participant_folders = get_folder_list(path=path, absolute=True, sort=sort)
    # filter:
    if id_prefix is not None:
        participant_folders = [pf for pf in participant_folders if pf.name.startswith(id_prefix)]
    if absolute:
        return participant_folders
    return [pf.name for pf in participant_folders]


def save_dataframe(df: pd.DataFrame, path_filename: Path, **kwargs):
    match path_filename.suffix:
        case '.csv':
            df.to_csv(path_filename, sep=',', header=True, index=True, **kwargs)
        case '.html':
            df.to_html(path_filename, header=True, index=True, **kwargs)
        case '.pkl' | '.pickle':
            df.to_pickle(str(path_filename), **kwargs)
        case '.h5' | '.hdf5':
            df.to_hdf(str(path_filename), key='df', mode='w', **kwargs)
        case '.xlsx' | '.xls':
            df.to_excel(path_filename, header=True, **kwargs)
        case '.json':
            df.to_json(path_filename, orient='index', indent=2, **kwargs)
        case _:
            print(f'Filetype {path_filename.suffix} not supported')


def load_dataframe(path_filename: Path = None, **kwargs) -> pd.DataFrame | List[pd.DataFrame] | object:
    if not path_filename.exists():
        raise FileNotFoundError(f'File {path_filename} not found')
    match path_filename.suffix:
        case '.csv':
            return pd.read_csv(path_filename, **kwargs)
        case '.html':
            return pd.read_html(path_filename, index_col=0, **kwargs)[0]
        case '.pkl' | '.pickle':
            return pd.read_pickle(path_filename, **kwargs)
        case '.h5' | '.hdf5':
            return pd.read_hdf(path_filename, key='df', mode='r', **kwargs)
        case '.xlsx' | '.xls':
            return pd.read_excel(path_filename, **kwargs)
        case '.json':
            return pd.read_json(path_filename, orient='index', **kwargs)
        case _:
            print(f'Filetype {path_filename} not supported')
