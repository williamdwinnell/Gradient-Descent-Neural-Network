import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.normal(0, 0.1, (self.input_size, self.hidden_size))
        self.W2 = np.random.normal(0, 0.1, (self.hidden_size, self.output_size))
        self.iteration = 0

    def forward(self, x):
        # forward propagation
        self.z = np.dot(x, self.W1) # dot product of input and first set of weights
        self.h = self.sigmoid(self.z) # activation function
        self.z2 = np.dot(self.h, self.W2) # dot product of hidden layer and second set of weights
        y_hat = self.sigmoid(self.z2) # final activation function
        return y_hat

    def sigmoid(self, z):
        # activation function
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        # derivative of sigmoid function
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def update_weights(self, x, y, y_hat, learning_rate):
        # update weights using gradient descent
        delta = y - y_hat
        d_W2 = np.dot(self.h.T, delta * self.sigmoid_prime(self.z2))
        d_W1 = np.dot(x.T, np.dot(delta * self.sigmoid_prime(self.z2), self.W2.T) * self.sigmoid_prime(self.z))

        # use a dynamic learning rate that decreases as the network converges
        self.W1 += learning_rate / (1 + self.iteration / 1000) * d_W1 + np.random.normal(0, 0.1, (self.input_size, self.hidden_size))
        self.W2 += learning_rate / (1 + self.iteration / 1000) * d_W2 + np.random.normal(0, 0.1, (self.hidden_size, self.output_size))

        # increment the iteration counter
        self.iteration += 1

    def train(self, x, y, epochs, learning_rate):
        # train the neural network
        for i in range(epochs):
            y_hat = self.forward(x)
            self.update_weights(x, y, y_hat, learning_rate)

# define the input and output data
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [0], [0], [1], [1], [1], [1], [1]])

# create a neural network with 2 input nodes, 4 hidden nodes, and 1 output node
nn = NeuralNetwork(len(x[0]), 5, 1)

# train the neural network for 2000 epochs with a learning rate of 0.1
nn.train(x, y, 300, 2)

print("Input: [0, 0, 0]\nPredicted output: ", nn.forward([0, 0, 0]), "\nActual output: [0]")
print("Input: [0, 0, 1]\nPredicted output: ", nn.forward([0, 0, 1]), "\nActual output: [0]")
print("Input: [0, 1, 0]\nPredicted output: ", nn.forward([0, 1, 0]), "\nActual output: [0]")
print("Input: [0, 1, 1]\nPredicted output: ", nn.forward([0, 1, 1]), "\nActual output: [1]")
print("Input: [1, 0, 0]\nPredicted output: ", nn.forward([1, 0, 0]), "\nActual output: [1]")
print("Input: [1, 0, 1]\nPredicted output: ", nn.forward([1, 0, 1]), "\nActual output: [1]")
print("Input: [1, 1, 0]\nPredicted output: ", nn.forward([1, 1, 0]), "\nActual output: [1]")
print("Input: [1, 1, 1]\nPredicted output: ", nn.forward([1, 1, 1]), "\nActual output: [1]")