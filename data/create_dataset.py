import os
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

def split_dataset(label_counts, val_split=.2):
    print(label_counts)
    train_indices, test_indices = [], []
    for i in label_counts:
        shuffled_indices = np.random.permutation(i)
        tr, ts = train_test_split(shuffled_indices, test_size=val_split, random_state=42)
        train_indices.append(tr)
        test_indices.append(ts)
        
    return train_indices, test_indices
        
def create_dataset(path, result_path, cell_types, times, val_split=.2):
    tags = sorted(cell_types)
    labels = list(range(len(tags)))
    
    if type(times) != list:
        times = list(times)
    
    save_counts = []
    train_indices, test_indices = [], []
    for time in times:
        datasets = []
        for tp in tags:
            data = []
            res_max_path = os.path.join(path, f'{tp}_max_signals.csv')
            if os.path.exists(res_max_path):
                fp = open(res_max_path, 'r')
                reader = csv.reader(fp)
                for line in reader:
                    if len(line) >= time:
                        line = list(map(float, line))[:time]
                        data.append(line)
            if len(data) == 0 or not os.path.exists(res_max_path):
                raise ValueError(f'Data does not exist for cell type {tp}')
            datasets.append(data)
        label_counts = list(map(len, datasets))
        if label_counts != save_counts:
            save_counts = label_counts
            train_indices, test_indices = split_dataset(label_counts, val_split)
        X_train, y_train, X_test, y_test = [], [], [], []
        for d, l, tr, ts in zip(datasets, labels, train_indices, test_indices):
            X_train.extend(list(np.array(d)[tr]))
            y_train.extend([l] * len(tr))
            X_test.extend(list(np.array(d)[ts]))
            y_test.extend([l] * len(ts))
        
        print(Counter(y_train), Counter(y_test))
        
        res_path = os.path.join(result_path, str(int(time * 9 / 60)))
        
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
        pd.DataFrame(X_train).to_csv(os.path.join(res_path, 'X_train.csv'), header=None, index=None)
        pd.DataFrame(y_train).to_csv(os.path.join(res_path,'y_train.csv'), header=None, index=None)
        pd.DataFrame(X_test).to_csv(os.path.join(res_path,'X_test.csv'), header=None, index=None)
        pd.DataFrame(y_test).to_csv(os.path.join(res_path,'y_test.csv'), header=None, index=None)
        pd.DataFrame(list(zip(tags, labels))).to_csv(os.path.join(res_path, 'dictionary.csv'), header=None, index=None)
        
        yield *list(map(np.array, [X_train, y_train, X_test, y_test])), (tags, labels)
    