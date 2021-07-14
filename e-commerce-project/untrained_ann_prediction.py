import numpy as np
from data_preprocess import get_data

X_train, y_train, X_test, y_test = get_data()

# We randomly initialise the weights: D input dims, M hidden layer size, K classes
D = X_train.shape[1]
M = 5
K = len(set(y_train))
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1) # can also have tanh or relu (with np.maximum)
    A = Z.dot(W2) + b2
    return softmax(A)

def classification_accuracy(Y, P):
    # We calculate the classification accuracy
    return np.mean(Y == P)

P_Y_given_X = forward(X_train, W1, b1, W2, b2)
# print("P_Y_given_X.shape:", P_Y_given_X.shape)
predictions = np.argmax(P_Y_given_X, axis=1)

print("Classification accuracy for randomly chosen weights:", classification_accuracy(y_train, predictions))
