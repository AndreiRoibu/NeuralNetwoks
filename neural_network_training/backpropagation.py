import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1) # can also have tanh or relu (with np.maximum)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def classification_rate(Y, P):
    # We calculate the classification rate as Ncorrect/Ntotal
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct +=1
    return float(n_correct) / n_total

def cost(T, Y):
    cost_value = T * np.log(Y)
    return cost_value.sum()

def derivative_w2(Z, T, Y_hat):
    return Z.T.dot(T-Y_hat)

def derivative_b2(T,Y_hat):
    return (T - Y_hat).sum(axis=0)

def derivative_w1(X, Z, T, Y_hat, W2):
    dZ = (T - Y_hat).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)

def derivative_b1(Z, T, Y_hat, W2):
    return ((T - Y_hat).dot(W2.T) * Z * (1-Z)).sum(axis=0)


def main():
    Nclass = 500
    # We randomly initialise the weights: D input dims, M hidden layer size, K classes
    D = 2
    M = 3
    K = 3
    # We generate some Gaussian Clouds
    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])

    X = np.vstack([X1, X2, X3])
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    N = len(Y)

    # Turn Y into an indicator variable (either 0 or 1) for training
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.title("Data Visualisation")
    plt.show()

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        Y_hat, Z = forward(X, W1, b1, W2, b2)
        if epoch % 10 == 0:
            c = cost(T, Y_hat) # calculate the cost
            P = np.argmax(Y_hat, axis=1) # predictions
            r = classification_rate(Y, P) 
            print("Cost: {}, Classification Rate: {}".format(c, r))
            costs.append(c)

        # We do gradient Ascent (not Descent)
        gW2 = derivative_w2(Z, T, Y_hat)
        gb2 = derivative_b2(T, Y_hat)
        gW1 = derivative_w1(X, Z, T, Y_hat, W2)
        gb1 = derivative_b1(Z, T, Y_hat, W2)
        W2 += learning_rate * gW2
        b2 += learning_rate * gb2 
        W1 += learning_rate * gW1 
        b1 += learning_rate * gb1  

    plt.figure()
    plt.plot(costs)
    plt.title("Costs")
    plt.show()    

if __name__ == '__main__':
    main()