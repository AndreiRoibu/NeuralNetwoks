import numpy as np
import matplotlib.pyplot as plt
from utils import get_data

labels = [
    'Anger',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
    ]

def main():
    X, Y, _, _ = get_data()

    while True:
        for i in range(7):
            # Loop throug all 7 emotions, choose images that reflect an emotion and then select a random one
            x, y = X[Y==i], Y[Y==i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(labels[y[j]])
            plt.show()

        prompt = input('Quit? Enter Y:\n')
        if prompt.lower().startswith('y'):
            break


if __name__ == '__main__':
    main()