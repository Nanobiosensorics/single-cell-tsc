import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_dataset(path, resample=True, scale=True):
    X_train = np.array(pd.read_csv(os.path.join(path, 'X_train.csv'), header=None))
    y_train = np.array(pd.read_csv(os.path.join(path, 'y_train.csv'), header=None))
    X_test = np.array(pd.read_csv(os.path.join(path, 'X_test.csv'), header=None))
    y_test = np.array(pd.read_csv(os.path.join(path, 'y_test.csv'), header=None))
    dictionary = pd.read_csv(os.path.join(path, 'dictionary.csv'), header=None)
    tags, labels = list(dictionary.iloc[:, 0]), list(dictionary.iloc[:, 1])
    
    print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    if resample:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    print('train:', list(zip(np.unique(y_train, return_counts=True))))
    print('test:', list(zip(np.unique(y_test, return_counts=True))))
    
    train = list(zip(X_train, y_train))
    random.shuffle(train)
    X_train, y_train = zip(*train)
    
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        return *list(map(np.array, [X_train, y_train, X_test, y_test])), (tags, labels), scaler
    
    return *list(map(np.array, [X_train, y_train, X_test, y_test])), (tags, labels)
        
    
    
    