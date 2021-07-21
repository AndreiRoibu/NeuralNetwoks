import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def forward(X, W1, b1, W2, b2):
    """As we are doing binary classification, just do 2 sigmoids in a row"""
    Z = X.dot(W1) + b1
    # Z = Z * (Z > 0) # ReLU 
    # Z = np.tanh(Z) # Tanh  
    Z = 1 / (1 + np.exp( -Z )) # Sigmoid
    Z2 = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp( -Z2 ))
    return Y, Z

def get_log_likelihood(Y, pY):
    return np.sum(Y * np.log(pY) + (1 - Y) * np.log( 1 - pY ))

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def derivative_w2(Z, Y, pY):
    return (Y - pY).dot(Z)

def derivative_b2(Y, pY):
    return (Y - pY).sum()

def derivative_w1(X, Z, Y, pY, W2):
    nonlinearity = Z * (1 - Z) # for sigmoid
    # nonlinearity = 1 - Z * Z # for tanh
    # nonlinearity = (Z > 0) # for relu
    dZ = np.outer(Y - pY, W2) * nonlinearity
    return X.T.dot(dZ)

def derivative_b1(Z, Y, pY, W2):
    nonlinearity = Z * (1 - Z) # for sigmoid
    # nonlinearity = 1 - Z * Z # for tanh
    # nonlinearity = (Z > 0) # for relu
    dZ = np.outer(Y - pY, W2) * nonlinearity
    return dZ.sum(axis=0)

def test_xor():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.random.randn(5)
    W2 = np.random.randn(5)
    b2 = np.random.randn(1)

    log_likelihoods = []
    learning_rate = 1e-2
    regularization = 0.
    
    for i in range(30000):
        pY, Z = forward(X, W1, b1, W2, b2)
        log_likelihood = get_log_likelihood(Y, pY)
        log_likelihoods.append(log_likelihood)
        prediction = predict(X, W1, b1, W2, b2)
        error = np.abs(prediction - Y).mean()

        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)

        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)

        if i % 1000 == 0:
            print("Epoch:", i, "Log-Likelihood:", log_likelihood, "Classification Rate:", 1 - error)

    print("Final Classification Rate:", np.mean(prediction == Y))
    plt.figure()
    plt.plot(log_likelihoods)
    plt.show()

def test_donut():

    N = 1000
    R_inner = 5
    R_outer = 10
    R1 = np.random.randn(N//2) + R_inner
    theta1 = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta1)], [R1 * np.sin(theta1)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta2 = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta2)], [R2 * np.sin(theta2)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N//2) + [1]* (N//2))

    n_hidden = 8

    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)

    log_likelihoods = []
    learning_rate = 1e-3
    regularization = 0.2
    
    for i in range(3000):
        pY, Z = forward(X, W1, b1, W2, b2)
        log_likelihood = get_log_likelihood(Y, pY)
        log_likelihoods.append(log_likelihood)
        prediction = predict(X, W1, b1, W2, b2)
        error = np.abs(prediction - Y).mean()

        gW2 = derivative_w2(Z, Y, pY)
        gb2 = derivative_b2(Y, pY)
        gW1 = derivative_w1(X, Z, Y, pY, W2)
        gb1 = derivative_b1(Z, Y, pY, W2)

        W2 += learning_rate * (gW2 - regularization * W2)
        b2 += learning_rate * (gb2 - regularization * b2)
        W1 += learning_rate * (gW1 - regularization * W1)
        b1 += learning_rate * (gb1 - regularization * b1)

        if i % 100 == 0:
            print("Epoch:", i, "Log-Likelihood:", log_likelihood, "Classification Rate:", 1 - error)

    print("Final Classification Rate:", np.mean(prediction == Y))
    plt.figure()
    plt.plot(log_likelihoods)
    plt.show()

if __name__ == '__main__':
    print("Testing XOR:")
    test_xor()
    print("Testing Donut:")
    test_donut()