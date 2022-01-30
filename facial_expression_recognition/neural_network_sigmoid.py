import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.utils import shuffle
from utils import get_binary_data, initialise_weights_biases, relu, sigmoid, sigmoid_cost, error_rate

class ANN(object):
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units

    def fit(self, Xtrain, Ytrain, learning_rate=1e-6, reg=1.0, epochs=10000, show_fig=False):
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        Xvalid, Yvalid = Xtrain[-1000:], Ytrain[-1000:]
        Xtrain, Ytrain = Xtrain[:-1000], Ytrain[:-1000]
        
        _, D = Xtrain.shape
        self.W1, self.b1 = initialise_weights_biases(D, self.hidden_units)
        self.W2, self.b2 = initialise_weights_biases(self.hidden_units, 1)

        costs = []
        best_validation_error = 1
        
        for i in range(epochs):
            
            # Forward propagation
            pY, Z = self.forward(Xtrain)

            # Gradient Descent
            deltaY = pY - Ytrain
            self.W2 -= learning_rate * (Z.T.dot(deltaY) + reg * self.W2)
            self.b2 -= learning_rate * ((deltaY).sum() + reg * self.b2)

            dZ = np.outer(deltaY, self.W2) * (Z > 0) # derivative of relu

            self.W1 -= learning_rate * (Xtrain.T.dot(dZ) + reg * self.W1)
            self.b1 -= learning_rate * (np.sum(dZ, axis=0) + reg * self.b1)

            if i % 20 == 0:
                pY_validation, _ = self.forward(Xvalid)
                cost = sigmoid_cost(Yvalid, pY_validation)
                costs.append(cost)
                erorr = error_rate(Yvalid, np.round(pY_validation))
                print("Epochs: {} | Cost: {} | Error Rate: {}".format(i, cost, erorr))
                if erorr < best_validation_error:
                    best_validation_error = erorr

        print("Best Validation Error: ", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.title('Costs')
            plt.xlabel("Epochs")
            plt.ylabel("Cost/Epoch")
            plt.show()

    def forward(self, X):
        Z = relu(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

def main():
    Xtrain, Ytrain, Xtest, Ytest = get_binary_data(Ntest=1000, balance_class_one=True)
    model = ANN(100)

    model.fit(Xtrain, Ytrain, show_fig=True)

    print("Test Score: ", model.score(Xtest, Ytest))

if __name__ == '__main__':
    main()


