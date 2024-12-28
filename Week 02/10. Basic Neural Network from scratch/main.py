import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error (MSE) Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Basic NN class
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    # Forward pass
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        #print('\nHidden input')
        #print(self.hidden_input)
        #print(self.hidden_input.shape)
        self.hidden_output = sigmoid(self.hidden_input)
        #print('\nHidden output')
        #print(self.hidden_output)
        #print(self.hidden_output.shape)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        #print('\noutput input')
        #print(self.output_input)
        #print(self.output_input.shape)
        self.output = sigmoid(self.output_input)
        #print('\noutput')
        #print(self.output)
        #print(self.output.shape)

        return self.output
    
    # Backward pass and weights update
    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, y, output, learning_rate)

            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")

# XOR datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = BasicNeuralNetwork(input_size=2, hidden_size=32, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained neural network
print("Test the trained neural network:")
for i in range(len(X)):
    predicted_output = nn.forward(X[i].reshape(1, -1))  # Ensure the input is 2D
    print(f"Input: {X[i]}, Predicted Output: {predicted_output[0][0]:.4f}, Actual Output: {y[i][0]}")
