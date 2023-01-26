import numpy as np
import math
import pandas as pd
def mean_var_over_Exps(list_exps, columns):
    a = dict()
    for column in columns[1:]:

        aggregate = np.array([np.array(exp[column]) for exp in list_exps])

        vec_mean = [np.mean(aggregate[:,j]) for j in range(aggregate.shape[1])]
        vec_var = [math.sqrt(np.var(aggregate[:,j])) for j in range(aggregate.shape[1])]
        a[column + '_mean'] = vec_mean
        a[column + '_var'] = vec_var
    return a


def find_unique_id_dictionary(ids_):
    inverse_id_dict = dict()
    out = np.unique(ids_)
    id_unique_dict = {out[i]: i for i in range(len(out))}
    for key, values in id_unique_dict.items():
        inverse_id_dict[values] = key

    return id_unique_dict, inverse_id_dict
