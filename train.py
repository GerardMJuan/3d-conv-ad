"""
Main function that trains the model from input data.

This function gathers the data and feeds tensors into the network, training it.
"""

from model import c3d

def main():
    # Load the model
    model = c3d.model()
    mean = c3d.mean

    model = vgg16.model(weights=True, summary=True)
    mean = vgg16.mean
    model.compile(loss='mse', optimizer='sgd')
    X = X - mean
    model.fit(X, Y)






if __name__ == '__main__':
    main()
