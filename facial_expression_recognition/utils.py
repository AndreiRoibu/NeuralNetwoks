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
    Xtest, Ytest = X[Ntest:], Y[Ntest:]

    if balance_class_one == True:
        Xoriginal, Yoriginal = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        Xone = Xtrain[Ytrain==1, :]
        Xone = np.repeat(Xone, 9, axis=0) # balancing class 1
        Xtrain = np.vstack([Xoriginal, Xone])
        Ytrain = np.concatenate((Yoriginal, [1*len(Xone)]))

    return Xtrain, Ytrain, Xtest, Ytest

def get_binary_data(Ntest=1000):
    file_path = directory_path + 'data/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv'

    # We read the data
    X, Y = [], []
    first_line = True
    for line in open(file_path):
        if first_line == True:
            first_line = False
        else:
            row = line.split(',')
            y = int(row([0]))
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])

    # We normalise the data in range [0,1]
    X, Y = np.array(X) / 255.0, np.array(Y)

    # We shuffle the data
    X, Y = shuffle(X, Y)

    # We split the data
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xtest, Ytest = X[Ntest:], Y[Ntest:]

    return Xtrain, Ytrain, Xtest, Ytest

if __name__ == '__main__':
    get_data()