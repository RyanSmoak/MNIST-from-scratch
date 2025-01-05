import numpy as np

class PReLU():
  def __init__(self, alpha=0.01):
    '''
      This function initializes the PReLU activation function. Seriously, initialize this stuff, makes life so much easier.
    '''
    self.alpha = alpha
    self.alpha_grad = None
    self.prelu_input = None

  def forward(self, prev_layer_input):
    '''
      This function does the forward pass of the PReLU activation function.
      Input:
        prev_layer_input: The input from the previous layer, be it the conv2D or the FC layer.
      Output:
        The output of the PReLU activation function.
    '''
    #storing the input with self to use in the backprop function
    self.prelu_input = prev_layer_input
    return np.where(prev_layer_input > 0, prev_layer_input, self.alpha * prev_layer_input)

  def backward(self, dy):
    '''
      This function does the backward pass of the PReLU activation function.
      Input:
        dy: The gradient of the loss function with respect to the output of the PReLU activation function.
      Output:
        dx: The gradient of the loss function with respect to the input of the PReLU activation function.
    '''
    if self.prelu_input is None:
      raise ValueError("Input to PReLU activation function has not been computed yet.")

    # Gradient of the activation with respect to the input
    dx = np.where(self.prelu_input > 0, dy, self.alpha * dy)

    # Gradient of alpha: sum of the gradients where input <= 0
    self.alpha_grad = np.sum(dy * self.prelu_input * (self.prelu_input <= 0))

    return dx

class Softmax():
  def __init__(self):
    self.output = None

  def softmax(self, logits):
    '''
    This function is compute the class probabilites for a certain data point
    using softmax activation function
    Input:
      x: The input array of size (batch_size, num_classes)
    Output:
      The class probabilites for the data point
    '''
    self.logits = logits
    exp_x = np.exp(self.logits - np.max(self.logits, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
