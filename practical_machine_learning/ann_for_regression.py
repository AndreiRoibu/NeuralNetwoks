import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

def forward(X, W1, b1, W2, b2):
    Z = X.dot(W1) + b1
    Z = Z * (Z > 0) # ReLU
    Yhat = Z.dot(W2) + b2
    return Z, Yhat

def derivative_W2(Z, Y, Yhat):
    return (Y-Yhat).dot(Z)

def derivative_b2(Y, Yhat):
    return (Y-Yhat).sum()

def derivative_W1(X, Z, Y, Yhat, W2):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    # or
    # nonlinearity = Z * (1 - Z) # for sigmoid
    # nonlinearity = 1 - Z * Z # for tanh
    # nonlinearity = (Z > 0) # for relu
    # dZ = np.outer(Y - Yhat, W2) * nonlinearity
    dZ = np.outer(Y-Yhat, W2) * (Z > 0)
    return X.T.dot(dZ)

def derivative_b1(Z, Y, Yhat, W2):
    # dZ = np.outer(Y - Yhat, V) * (1 - Z * Z) # this is for tanh activation
    dZ = np.outer(Y-Yhat, W2) * (Z > 0)
    return dZ.sum(axis=0)

def update(X, Z, Y, Yhat, W1, W2, b1, b2, learning_rate=1e-4, regularization=0):
    gW2 = derivative_W2(Z, Y, Yhat)
    gb2 = derivative_b2(Y, Yhat)
    gW1 = derivative_W1(X, Z, Y, Yhat, W2)
    gb1 = derivative_b1(Z, Y, Yhat, W2)

    W2 += learning_rate * (gW2 - regularization * W2)
    b2 += learning_rate * (gb2 - regularization * b2)
    W1 += learning_rate * (gW1 - regularization * W1)
    b1 += learning_rate * (gb1 - regularization * b1)

    return W1, b1, W2, b2

def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()

if __name__ == '__main__':

    # Generate the dataset
    N = 500
    X = np.random.random((N, 2)) * 4 - 2
    Y = X[:,0] * X[:, 1] # we make a saddle

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:, 1], Y)
    plt.show()

    # MAKE a NN and train it
    dimensionality = 2
    hidden_units = 100
    W1 = np.random.randn(dimensionality, hidden_units) / np.sqrt(dimensionality)
    b1 = np.zeros(hidden_units)
    W2 = np.random.randn(hidden_units) / np.sqrt(hidden_units)
    b2 = 0.0

    # TRAIN the NN:
    costs = []
    for i in range(200):
        Z, Yhat = forward(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = update(X, Z, Y, Yhat, W1, W2, b1, b2)
        cost = get_cost(Y, Yhat)
        costs.append(cost)
        if i % 25 == 0:
            print("Epoch:", i, "Cost:", cost)

    plt.figure()
    plt.plot(costs)
    plt.show()

    # PLOT the prediction with the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    # surface plot
    line = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    _, Yhat = forward(Xgrid, W1, b1, W2, b2)
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True, alpha=0.5)
    plt.show()
        
    # PLOT magnitude of the residuals
    Ygrid = Xgrid[:, 0] * Xgrid[:, 1]
    R = np.abs(Ygrid - Yhat)
    plt.figure()
    plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
    plt.show()
