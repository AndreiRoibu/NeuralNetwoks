import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.utils import shuffle
from utils import get_data, initialise_weights_biases, sigmoid, sigmoid_cost, error_rate
from utils import y_to_indicator, softmax, softmax_cost

class ANN_Logistic(object):
    def __init__(self):
        pass

    def fit(self, Xtrain, Ytrain, learning_rate=1e-7, reg=0.0, epochs=10000, show_fig=False):
       
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        Xvalid, Yvalid = Xtrain[-1000:], Ytrain[-1000:]
        Xtrain, Ytrain = Xtrain[:-1000], Ytrain[:-1000]
        
        Ttrain = y_to_indicator(Ytrain)
        Tvalid = y_to_indicator(Yvalid)

        _, D = Xtrain.shape
        classes = len(set(Ytrain))
        self.W, self.b = initialise_weights_biases(D, classes)
        
        costs = []
        best_validation_error = 1
        
        for i in range(epochs):
            
            # Forward propagation
            pY = self.forward(Xtrain)

            # Gradient Descent
            deltaY = pY - Ttrain
            self.W -= learning_rate * (Xtrain.T.dot(deltaY) + reg * self.W)
            self.b -= learning_rate * ((deltaY).sum(axis=0) + reg * self.b)

            if i % 20 == 0:
                pY_validation = self.forward(Xvalid)
                cost = sigmoid_cost(Tvalid, pY_validation)
                costs.append(cost)
                erorr = error_rate(Tvalid, np.round(pY_validation))
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
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


class ANN(object):
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units

    def fit(self, Xtrain, Ytrain, learning_rate=1e-7, reg=0.0, epochs=10000, show_fig=False):
        
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        Xvalid, Yvalid = Xtrain[-1000:], Ytrain[-1000:]
        Xtrain, Ytrain = Xtrain[:-1000], Ytrain[:-1000]
        
        Ttrain = y_to_indicator(Ytrain)

        _, D = Xtrain.shape
        classes = len(set(Ytrain))
        self.W1, self.b1 = initialise_weights_biases(D, self.hidden_units)
        self.W2, self.b2 = initialise_weights_biases(self.hidden_units, classes)
        
        costs = []
        best_validation_error = 1
        
        for i in range(epochs):
            
            # Forward propagation
            pY, Z = self.forward(Xtrain)

            # Gradient Descent
            deltaY = pY - Ttrain
            self.W2 -= learning_rate * (Z.T.dot(deltaY) + reg * self.W2)
            self.b2 -= learning_rate * ((deltaY).sum(axis=0) + reg * self.b2)

            dZ = deltaY.dot(self.W2.T) * (1 - Z*Z) # derivative of tanh

            self.W1 -= learning_rate * (Xtrain.T.dot(dZ) + reg * self.W1)
            self.b1 -= learning_rate * ( dZ.sum(axis=0) + reg * self.b1 )

            if i % 20 == 0:
                pY_validation, _ = self.forward(Xvalid)
                cost = softmax_cost(Yvalid, pY_validation)
                costs.append(cost)
                erorr = error_rate(Yvalid, np.argmax(pY_validation, axis=1))
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
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

def main():
    Xtrain, Ytrain, Xtest, Ytest = get_data(Ntest=1000, balance_class_one=True)
    
    # model = ANN_Logistic()
    model = ANN(200)

    model.fit(Xtrain, Ytrain, show_fig=True)

    print("Test Score: ", model.score(Xtest, Ytest))

if __name__ == '__main__':
    main()