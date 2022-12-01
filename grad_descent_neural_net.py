import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    '''def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))'''
    
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
 
    def average_error(self, x, y):
        # initialize the total error to 0
        total_error = 0

        # loop over the input-output pairs
        for i in range(len(x)):
            # calculate the predicted output
            predicted_output = self.forward(x[i])

            # calculate the error for this pair
            error = np.mean((predicted_output - y[i]) ** 2)

            # add the error to the total error
            total_error += error

        # divide the total error by the number of input-output pairs
        average_error = total_error / len(x)

        return average_error

###MAIN CODE###

# load the CSV file using Pandas
df = pd.read_csv(r'C:\Users\liamd\OneDrive\Desktop\Projects\Optimizers_From_Scratch\housepricedata.csv')

# select the columns to use as input and output data
x = df[["LotArea", "OverallQual", "OverallCond", "TotalBsmtSF", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageArea"]]
y = df[["AboveMedianPrice"]]
y = y.to_numpy()[:1000]
x = x.to_numpy()[:1000]

#create a StandardScaler object
scaler = StandardScaler()

# fit the scaler to the input data
scaler.fit(x)

# transform the input data using the scaler
x = scaler.transform(x)

# create a neural network with the same number of input and output nodes as the input and output data
nn = NeuralNetwork(x.shape[1], 4, y.shape[1])

#train for 3000 epochs with an initial learning rate of 2
nn.train(x, y, 3000, 2)

for i in range(3):
    input_data = x[i]
    actual_output = y[i]
    predicted_output = nn.forward(input_data)
    print("Input:", input_data, "\nPredicted output:", predicted_output, "\nActual output:", actual_output,'\n')

print("Mean Error: ", nn.average_error(x, y))

'''
#Here is another simpler example using a smaller dataset.

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

print("Mean Error: ", nn.average_error(x, y))
'''