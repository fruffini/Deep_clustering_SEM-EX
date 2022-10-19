import numpy as np



def find_unique_id_dictionary(ids_):
    inverse_id_dict = dict()
    out = np.unique(ids_)
    id_unique_dict = {out[i]: i for i in range(len(out))}
    for key, values in id_unique_dict.items():
        inverse_id_dict[values] = key

    return id_unique_dict, inverse_id_dict
