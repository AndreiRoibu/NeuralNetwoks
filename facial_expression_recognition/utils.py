import numpy as np
from sklearn.utils import shuffle
import os

# To allow scipts from other folders to import this file:
directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
directory_path = directory_path.replace('NeuralNetwoks/facial_expression_recognition', '')

def get_data(Ntest=1000, balance_class_one=False):
    file_path = directory_path + 'data/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv'

    # We read the data
    X, Y = [], []
    first_line = True
    for line in open(file_path):
        if first_line == True:
            first_line = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    # We normalise the data in range [0,1]
    X, Y = np.array(X) / 255.0, np.array(Y)

    # We shuffle the data
    X, Y = shuffle(X, Y)

    # We split the data
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xtest, Ytest = X[-Ntest:], Y[-Ntest:]

    if balance_class_one == True:
        Xoriginal, Yoriginal = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        Xone = Xtrain[Ytrain==1, :]
        Xone = np.repeat(Xone, 9, axis=0) # balancing class 1
        Xtrain = np.vstack([Xoriginal, Xone])
        Ytrain = np.concatenate((Yoriginal, [1*len(Xone)]))

    return Xtrain, Ytrain, Xtest, Ytest

def get_binary_data(Ntest=1000, balance_class_one=False):
    file_path = directory_path + 'data/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv'

    # We read the data
    X, Y = [], []
    first_line = True
    for line in open(file_path):
        if first_line == True:
            first_line = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])

    # We normalise the data in range [0,1]
    X, Y = np.array(X) / 255.0, np.array(Y)

    # We shuffle the data
    X, Y = shuffle(X, Y)

    # We split the data
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xtest, Ytest = X[-Ntest:], Y[-Ntest:]

    if balance_class_one == True:
        Xoriginal = Xtrain[Ytrain!=1, :]
        Xone = Xtrain[Ytrain==1, :]
        Xone = np.repeat(Xone, 9, axis=0) # balancing class 1
        Xtrain = np.vstack([Xoriginal, Xone])
        Ytrain = np.array([0] * len(Xoriginal) + [1] * len(Xone))

    return Xtrain, Ytrain, Xtest, Ytest

def initialise_weights_biases(input_size, output_size):
    if output_size != 1:
        W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        b = np.zeros(output_size)
        return W.astype(np.float32), b.astype(np.float32)
    else:
        W = np.random.randn(input_size) / np.sqrt(input_size)
        b = 0
        return W.astype(np.float32), b
    

def relu(X):
    return X * (X > 0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_cost(T, Y):
    return - (T * np.log(Y) + (1-T) * np.log(1 - Y)).sum()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def y_to_indicator(y):
    N = len(y)
    K = len(set(y))
    indicator = np.zeros((N, K))
    for i in range(N):
        indicator[i, y[i]] = 1
    return indicator

if __name__ == '__main__':
    get_binary_data()