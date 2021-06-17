#!/usr/bin/env python
from attack_utils import load_dataset
import pandas as pd

def statistics(mode): 
    df = pd.DataFrame(load_dataset(mode)[1])
    return df.sum(axis=0).tolist() + [df.shape[0]]

def latex_pretty(stat_list): 
    return " & ".join(stat_list)

if __name__ == "__main__": 
    for mode in ["train", "test"]: 
        print(latex_pretty([str(x) for x in statistics(mode)]))