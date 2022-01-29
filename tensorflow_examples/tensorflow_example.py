import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def initialise_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

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

    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])

    W1 = initialise_weights([D, M])
    b1 = initialise_weights([M])
    W2 = initialise_weights([M, K])
    b2 = initialise_weights([K])

    logits = forward(tfX, W1, b1, W2, b2)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tfY,
            logits=logits
        )
    )

    train_operation = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # learning_rate = 0.05

    predict_operation = tf.argmax(logits, 1) #choose the max on axis=1

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        sess.run(train_operation, feed_dict={tfX: X, tfY: T})
        pred = sess.run(predict_operation, feed_dict={tfX: X, tfY: T})
        if i % 100 == 0:
            print("Accuracy:", np.mean(Y == pred))


if __name__ == '__main__':
    main()