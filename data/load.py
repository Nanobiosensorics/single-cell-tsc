import os
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def repeat_list_to_length(lst, desired_length):
    repeated_list = []
    while len(repeated_list) < desired_length:
        repeated_list.extend(lst)
    return repeated_list[:desired_length]

def split_dataset(datasets, types, val_split=.2, upsample=True):
    X_train, y_train, X_test, y_test = [], [], [], []
    print(list(map(len, datasets)))
    max_len = int(max(list(map(len, datasets))) * (1 - val_split))
    
    for t, s in zip(types, datasets):
        shuffled_indices = np.random.permutation(len(s))
        train_indices, test_indices = train_test_split(shuffled_indices, test_size=val_split, random_state=42)
        train = list(np.array(s)[train_indices])
        # train_extended = repeat_list_to_length(train, max_len) if upsample else train
        train_extended = resample(train, replace=True, n_samples=max_len, random_state=42) if upsample else train
        X_train.extend(train_extended)
        y_train.extend([t] * len(train_extended))
        X_test.extend(list(np.array(s)[test_indices]))
        y_test.extend([t] * len(test_indices))

    train = list(zip(X_train, y_train))
    random.shuffle(train)
    X_train, y_train = zip(*train)
    return list(map(np.array, [X_train, y_train, X_test, y_test])) 
        
def create_dataset(path, cell_types, time, val_split=.2, upsample=True, scale=True):
    tags = sorted(cell_types)
    labels = list(range(len(tags)))
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
    
    X_train, y_train, X_test, y_test = split_dataset(datasets, labels, val_split, upsample)
    
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, (tags, labels)
    