"""
A utility to store and load tensors in hdf5.
"""


import h5py
import numpy
import torch
import termcolor
import logging
import json
from pathlib import Path
from collections import namedtuple


METADATA_DIR = Path("~/.invresnet-metadata/").expanduser()
DEFAULT_METADATA_FILE = METADATA_DIR/"metadata.h5"


def save_data(fname = DEFAULT_METADATA_FILE, **kwargs):
    fname.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    logging.info("saving metadata to %s", termcolor.colored(str(fname), "green"))

    metadata = {
        ds_name: isinstance(ds_data, (torch.Tensor, numpy.ndarray))
        for ds_name, ds_data in kwargs.items()
    }

    with h5py.File(fname, 'w') as h5ds:
        for ds_name, data in kwargs.items():
            if hasattr(data, 'to'):
                data = data.to('cpu')
            elif hasattr(data, 'cpu'):
                data = data.cpu()

            if hasattr(data, 'numpy'):
                data = data.numpy()

            try:
                h5ds.create_dataset(ds_name, data=data)
            except Exception:
                logging.error("failed to serialize variable %s, %r", ds_name, data)

        h5ds.create_dataset("__save_metadata", data=json.dumps(metadata))


def adjust_type(data: numpy.ndarray, cuda: bool, was_tensor: bool, to_torch: bool):
    if not isinstance(data, numpy.ndarray):
        return data

    if not was_tensor:
        return list(data)

    if not to_torch:
        return data

    data = torch.from_numpy(data)
    return data.cuda() if cuda else data



def load_data(fname = DEFAULT_METADATA_FILE, cuda=False, to_torch=True):
    with h5py.File(fname, 'r') as h5ds:
        metadata = json.loads(h5ds['__save_metadata'][()])
        loaded_data = [
            (
                ds_name,
                adjust_type(
                    ds_data[()],
                    cuda,
                    metadata.get(ds_name),
                    to_torch,
                ),
            )
            for ds_name, ds_data in h5ds.items()
            if ds_name != "__save_metadata"
        ]

    keys = [x for x, _ in loaded_data]
    values = [y for _, y in loaded_data]
    return namedtuple('LOADED_DATA', keys)(*values)
