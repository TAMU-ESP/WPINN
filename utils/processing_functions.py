import numpy as np
import pandas as pd

def get_out_of_dist_split_indexes(BP, BP_type='SBP', percentile=5):
    '''
    Function for splitting waveforms according to percentile
    '''
    sub_length = len(BP)
    if BP_type == 'SBP':
        BP = np.max(BP, axis=-1)
    elif BP_type == 'MAP':
        BP = np.mean(BP, axis=-1)
    elif BP_type == 'DBP':
        BP = BP[:, 0]
    bp_min, bp_max = np.percentile(BP, percentile), np.percentile(BP, 100 - percentile)
    test_ind = np.where(np.logical_and(BP > bp_min,
                                       BP < bp_max))[0]
    train_ind = np.where(~np.isin(np.arange(sub_length), test_ind))[0]
    val_ind = train_ind[::5]
    train_ind = train_ind[np.where(~np.isin(train_ind, val_ind))[0]]
    return train_ind, val_ind, test_ind

def moving_average(data, w=10):
    '''
    Function for applying a moving average filter
    '''
    return pd.Series(data).rolling(window=w, min_periods=1, center=True).mean().to_numpy()