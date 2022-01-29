import numpy as np
from sklearn.utils import shuffle
import os

# To allow scipts from other folders to import this file:
directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
directory_path = directory_path.replace('NeuralNetwoks/facial_expression_recognition', '')

def get_data(Ntest=1000):
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

    # We shuffle the data
    X, Y = shuffle(X, Y)

    # We split the data
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xtest, Ytest = X[Ntest:], Y[Ntest:]

    return Xtrain, Ytrain, Xtest, Ytest

if __name__ == '__main__':
    get_data()