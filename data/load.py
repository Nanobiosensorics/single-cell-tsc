import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

def _load_data(path):
    X_train = np.array(pd.read_csv(os.path.join(path, 'X_train.csv'), header=None))
    y_train = np.array(pd.read_csv(os.path.join(path, 'y_train.csv'), header=None))
    X_test = np.array(pd.read_csv(os.path.join(path, 'X_test.csv'), header=None))
    y_test = np.array(pd.read_csv(os.path.join(path, 'y_test.csv'), header=None))
    dictionary = pd.read_csv(os.path.join(path, 'dictionary.csv'), header=None)
    tags, labels = list(dictionary.iloc[:, 0]), list(dictionary.iloc[:, 1])
    # print(y_train.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, tags, labels

def load_dataset(path, resample=True, scale=True):
    X_train, y_train, X_test, y_test, tags, labels = _load_data(path)
    
    # print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    if resample:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = y_train.reshape(-1, 1)
    
    # print('train:', list(zip(np.unique(y_train, return_counts=True))))
    # print('test:', list(zip(np.unique(y_test, return_counts=True))))
    
    train = list(zip(X_train, y_train))
    random.shuffle(train)
    X_train, y_train = zip(*train)
    X_train, y_train = np.array(X_train), np.array(y_train)
    scaler = None
    
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
    return X_train, y_train, X_test, y_test, (tags, labels), scaler

def load_dataset_kfold(path, kfold=5, resample=True, scale=True):
    X_train, y_train, X_test, y_test, tags, labels = _load_data(path)
    
    # print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    X_data, y_data = np.vstack([X_train, X_test]), np.vstack([y_train, y_test])
    
    # print('data:', list(zip(np.unique(y_data, return_counts=True))))
    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=52)
    
    for i, (train_index, test_index) in enumerate(skf.split(X_data, y_data)):
        X_train, y_train, X_test, y_test = X_data[train_index], y_data[train_index], X_data[test_index], y_data[test_index]
        
        if resample:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            y_train = y_train.reshape(-1, 1)
        
        # print()
        # print(i, 'train:', list(zip(np.unique(y_train, return_counts=True))))
        # print(i, 'test:', list(zip(np.unique(y_test, return_counts=True))))
            
        scaler = None
        
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        yield X_train, y_train, X_test, y_test, (tags, labels), scaler