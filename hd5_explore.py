import torch
import h5py as h5
import numpy as np
import pandas as pd

def hdf_read(filename):
    with h5.File(filename, "r") as f:
        list_ligand,list_label = [],[]
        base_items = list(f.keys())
        for item in base_items:
            ligand = f.get(item)

            ligand_data = np.array(ligand.get('ligand'))
            ligand_label = ligand.attrs['label']
            list_ligand.append(ligand_data)
            list_label.append(ligand_label)


    ligand_arr = np.array(list_ligand)
    label_arr = np.array(list_label)
    return torch.Tensor(ligand_arr),torch.Tensor(np.expand_dims(label_arr, axis=1))


