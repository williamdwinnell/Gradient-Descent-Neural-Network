import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.normal(0, 0.2, (self.input_size, self.hidden_size))
        self.W2 = np.random.normal(0, 0.2, (self.hidden_size, self.output_size))
        self.b1 = np.random.normal(0, 0.2, (1, self.hidden_size))
        self.b2 = np.random.normal(0, 0.2, (1, self.output_size))

        self.iteration = 0

    def forward(self, x):
        # forward propagation
        self.z = np.dot(x, self.W1) + self.b1 # dot product of input and first set of weights, with bias
        self.h = self.sigmoid(self.z) # activation function
        self.z2 = np.dot(self.h, self.W2) + self.b2 # dot product of hidden layer and second set of weights, with bias
        y_hat = self.sigmoid(self.z2) # final activation function
        return y_hat

    def sigmoid(self, z):

        # clip z to avoid overflow errors
        z = np.clip(z, -100, 100)

        # activation function
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):

        # clip z to avoid overflow errors
        z = np.clip(z, -100, 100)

        # derivative of sigmoid function
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    
    def update_weights(self, x, y, y_hat, learning_rate):
        # update weights using gradient descent
        delta = y - y_hat
        d_W2 = np.dot(self.h.T, delta * self.sigmoid_prime(self.z2))
        d_W1 = np.dot(x.T, np.dot(delta * self.sigmoid_prime(self.z2), self.W2.T) * self.sigmoid_prime(self.z))

        d_b2 = np.sum(delta * self.sigmoid_prime(self.z2), axis=0, keepdims=True)
        d_b1 = np.sum(np.dot(delta * self.sigmoid_prime(self.z2), self.W2.T) * self.sigmoid_prime(self.z), axis=0, keepdims=True)

        # use a dynamic learning rate that decreases as the network converges
        self.W1 += learning_rate / (1 + self.iteration / 1000) * d_W1 + np.random.normal(0, 0.1, (self.input_size, self.hidden_size))
        self.W2 += learning_rate / (1 + self.iteration / 1000) * d_W2 + np.random.normal(0, 0.1, (self.hidden_size, self.output_size))

        # update hidden layer bias weights
        self.b1 += learning_rate / (1 + self.iteration / 1000) * d_b1
        self.b2 += learning_rate / (1 + self.iteration / 1000) * d_b2

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

def find_optimal_learning_rate(x, y, network, learning_rates, num_tests):

    # store initial parameters
    temp_input_size = network.input_size
    temp_hidden_size = network.hidden_size
    temp_output_size = network.output_size

    # define the number of epochs to use for testing
    epochs = 15

    # create a dictionary to store the results
    results = {}

    # loop through the learning rates
    for learning_rate in learning_rates:
        # create a list to store the results for each test
        result_list = []

        # loop through the number of tests
        for i in range(num_tests):

            network.__init__(temp_input_size, temp_hidden_size, temp_output_size)

            # train the network with the current learning rate
            network.train(x, y, epochs, learning_rate)

            # evaluate the network on the training data
            y_hat = network.forward(x)

            # calculate the mean squared error
            mse = np.mean((y - y_hat) ** 2)

            # append the result to the list
            result_list.append(mse)

        # calculate the average mean squared error
        avg_mse = np.mean(result_list)

        # store the result in the dictionary
        results[learning_rate] = avg_mse
        print(learning_rate, ": ", avg_mse)
    # return the learning rate with the minimum average mean squared error
    return min(results, key=results.get)

###MAIN CODE###

# load the CSV file using Pandas
df = pd.read_csv(r'C:\Users\liamd\OneDrive\Desktop\Projects\Optimizers_From_Scratch\housepricedata.csv')

# select the columns to use as input and output data
x = df[["LotArea", "OverallQual", "OverallCond", "TotalBsmtSF", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageArea"]]
y = df[["AboveMedianPrice"]]

# split the data into train and test
y_train = y.to_numpy()[:1000]
x_train = x.to_numpy()[:1000]
y_test = y.to_numpy()[1000:1100]
x_test = x.to_numpy()[1000:1100]

# create a StandardScaler object
scaler = StandardScaler()

# fit the scaler to the input data
scaler.fit(x_train)

# transform the input data using the scaler
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# create a neural network with the same number of input and output nodes as the input and output data
nn = NeuralNetwork(x_train.shape[1], 3, y_train.shape[1])

# test different learning rates to find the optimal starting learning rate
learning_rate = find_optimal_learning_rate(x_train, y_train, nn, [1, 0.1, 0.01, 0.001], 10)
print(learning_rate)

# train for n epochs with an initial learning rate set by the find_optimal_learning_rate function
nn.train(x_train, y_train, 2000, learning_rate)

# print the final mse to see how the model performed 
print("Mean Error: ", nn.average_error(x_test, y_test))

'''
# Example usage of predicting an input with the model, plus a comparison to the actuals
for i in range(3):
    input_data = x[i]
    actual_output = y[i]
    predicted_output = nn.forward(input_data)
    print("Input:", input_data, "\nPredicted output:", predicted_output, "\nActual output:", actual_output,'\n')
'''