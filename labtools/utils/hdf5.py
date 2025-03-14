import h5py
import numpy as np


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if item is None:
            grp = h5file.create_group(path + key)
            grp.attrs['__is_none__'] = True
        elif isinstance(item, np.ndarray):
            if item.dtype.kind in {'U', 'S'}:
                # Handle arrays of strings
                dt = h5py.string_dtype(encoding='utf-8')
                # Convert the array to an array of objects
                item_as_object = item.astype('object')
                h5file.create_dataset(path + key, data=item_as_object, dtype=dt)
            else:
                # Handle other NumPy arrays (e.g., numeric arrays)
                h5file[path + key] = item
        elif isinstance(item, (np.number, int, float)):
            h5file[path + key] = item
        elif isinstance(item, str):
            # Save string as UTF-8 encoded variable-length string
            dt = h5py.string_dtype(encoding='utf-8')
            h5file.create_dataset(path + key, data=item, dtype=dt)
        elif isinstance(item, bytes):
            # Save bytes using h5py.vlen_dtype(bytes)
            dt = h5py.vlen_dtype(bytes)
            dset = h5file.create_dataset(path + key, data=item, dtype=dt)
            dset.attrs['is_bytes'] = True
        elif isinstance(item, dict):
            # Recursively save the dictionary
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise TypeError(f"Cannot save data type {type(item)} for key '{key}'")


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            if 'is_bytes' in item.attrs:
                ans[key] = item[()]
            elif h5py.check_string_dtype(item.dtype):
                # It's a string dataset
                data = item[()]
                if isinstance(data, np.ndarray):
                    # Data is an array of strings
                    data = data.astype(str)
                elif isinstance(data, bytes):
                    data = data.decode('utf-8')
                ans[key] = data
            else:
                ans[key] = item[()]  # Handles scalars and arrays
        elif isinstance(item, h5py.Group):
            if '__is_none__' in item.attrs and item.attrs['__is_none__']:
                ans[key] = None
            else:
                # Recursively load the group
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
        else:
            raise TypeError(f"Unknown item type {type(item)} for key '{key}'")
    return ans
