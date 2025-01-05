import numpy as np
from activations import PReLU

class FullyConnectedLayer:
    def __init__(self, input_neurons_num, output_neurons_num, activation_function, activation_derivative):
        '''
        Initialize the fully connected layer. No comment!
        Input:
          input_neurons_num: Number of input neurons.
          output_neurons_num: Number of output neurons.
          activation_function: The activation function to apply.
          activation_derivative: The derivative of the activation function.
        '''
        self.input_size = input_neurons_num
        self.output_size = output_neurons_num
        self.activation_function_instance = activation_function
        self.activation_derivative_instance = activation_derivative

        # Initialize weights and biases with He initialization
        self.weights = np.random.randn(output_neurons_num, input_neurons_num) * np.sqrt(2 / input_neurons_num)
        self.biases = np.zeros((output_neurons_num, 1))
        self.z = None
        self.activated_output = None

    def forward(self, input_data):
        '''
        Perform the forward pass through the FC layer.
        Input:
          input_data: Input data of shape (input_neurons_num, batch_size).
        Output:
          Output data after applying the activation function.
        '''
        input_data = input_data.reshape(self.input_size, -1)
        self.input = input_data  # Store input for use in backward pass
        self.z = np.dot(self.weights, input_data) + self.biases

        if self.activation_function_instance is not None:
          if isinstance(self.activation_function_instance, PReLU):
            self.activated_output = self.activation_function_instance.forward(self.z)
          else:
              self.activated_output = self.activation_function_instance(self.z)
          return self.activated_output
        else:
          self.activated_output = self.z

          return self.activated_output

    def backward(self, output_gradient, learning_rate):
        '''
        Perform the backward pass through the FC layer.
        Input:
          output_gradient: Gradient of the loss w.r.t. the output (a) of this layer.
          learning_rate: Learning rate for weight updates.
        Output:
          dx Gradient of the loss w.r.t. the input (x) of this layer.
        '''

        # Compute gradient w.r.t. z
        if self.activation_derivative_instance is not None:
          if isinstance(self.activation_function_instance, PReLU):
              dz = output_gradient * self.activation_derivative_instance.backward(self.activated_output)
          else:
              dz = output_gradient * self.activation_derivative_instance(self.activated_output)

          #dz = output_gradient * self.activation_derivative(self.activated_output)  # Element-wise multiplication
        else: dz = output_gradient

        # Compute gradients
        dw = np.dot(dz, self.input.T) / dz.shape[1]  # Weight gradient
        db = np.sum(dz, axis=1, keepdims=True) / dz.shape[1]  # Bias gradient

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db  # Broadcasting now works correctly

        # Compute input gradient
        dx = np.dot(self.weights.T, dz)
        return dx

