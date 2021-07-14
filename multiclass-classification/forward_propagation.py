import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

Nclass = 500
# We generate some Gaussian Clouds
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])

X = np.vstack([X1, X2, X3])
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.title("Data Visualisation")
plt.show()

# We randomly initialise the weights: D input dims, M hidden layer size, K classes
D = 2
M = 3
K = 3
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1) # can also have tanh or relu (with np.maximum)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

def classification_rate(Y, P):
    # We calculate the classification rate as Ncorrect/Ntotal
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct +=1
    return float(n_correct) / n_total

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

# Check that we have the correct axis above
assert(len(P) == len(Y))

print("Classification rate for randomly chosen weights:", classification_rate(Y, P))
