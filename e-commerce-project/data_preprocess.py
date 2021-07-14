import numpy as np
import pandas as pd
import os

# To allow scipts from other folders to import this file:
directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
directory_path = directory_path.replace('NeuralNetwoks/e-commerce-project', '')

def get_data():
    df = pd.read_csv(directory_path + 'large_files/ecommerce_data.csv')
    data = df.values # We extract the numerical values
    np.random.shuffle(data) # We shuffle the data

    # We split the data into features and labels
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int32)

    # We one-hot encode the data
    samples, features = X.shape
    X_new = np.zeros((samples, features + 3))
    X_new[:, 0:(features-1)] = X[:, 0:(features-1)] # Non-categorial data

    Z = np.zeros((samples, 4))
    Z[np.arange(samples), X[:, features-1].astype(np.int32)] = 1
    X_new[:, -4:] = Z # Categorial data

    X = X_new
    del X_new, Z
    
    # We split and normalize

    X_train = X[:-100]
    y_train = y[:-100]
    X_test = X[-100:]
    y_test = y[-100:]

    for i in (1, 2):
        mean = X_train[:,i].mean()
        std = X_train[:,i].std()
        X_train[:,i] = (X_train[:, i] - mean) / std
        X_test[:,i] = (X_test[:, i] - mean) / std

    return X_train, y_train, X_test, y_test

def get_binary_data():
    # Only returns the data from the first 2 classes
    X_train, y_train, X_test, y_test = get_data()
    return X_train[y_train <=1 ], y_train[y_train <=1 ], X_test[y_test <=1 ], y_test[y_test <=1 ]

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
    print(y_test)
    X_train, y_train, X_test, y_test = get_binary_data()
    print(y_test)