import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from data_preprocess import get_data

def y2indicator(Y, K):
    N = len(Y)
    indicator = np.zeros((N, K))
    for i in range(N):
        indicator[i, Y[i]] = 1
    return indicator

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=1, keepdims=True)

def cross_entropy(T, pY):
    return -np.mean( T * np.log(pY) )

def classification_rate(Y, P):
    return np.mean(Y == P)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    D = Xtrain.shape[1]
    K = len(set(Ytrain) | set(Ytest))
    M = 5 # Number of hidden units

    Ytrain_ind = y2indicator(Ytrain, K)
    Ytest_ind = y2indicator(Ytest, K)

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    train_costs = []
    test_costs = []
    learning_rate = 1e-3

    for i in range(10000):
        pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
        pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

        costTrain = cross_entropy(Ytrain_ind, pYtrain)
        costTest = cross_entropy(Ytest_ind, pYtest)
        train_costs.append(costTrain)
        test_costs.append(costTest)

        W2 -= learning_rate * Ztrain.T.dot(pYtrain - Ytrain_ind)
        b2 -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)
        
        dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain)
        W1 -= learning_rate * Xtrain.T.dot(dZ)
        b1 -= learning_rate * dZ.sum(axis=0)

        if i % 1000 == 0:
            print(i, costTrain, costTrain)

print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))

plt.figure()
plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()