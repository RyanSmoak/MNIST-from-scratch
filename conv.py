import numpy as np
from scipy import signal

class Conv2DLayer:
  '''
    This class creates the convolutional layer, the most basic and important layer in our CNN.
  '''
  def __init__(self, input_shape, filter_shape, stride, padding):
    '''
      This function initializes the convolutional layer.
      Inputs:
        - input_shape: The shape of the input structure (depth, height, width)
        - filter_shape: The shape of the filter (num_filters, depth, height, width)
        - stride: The number of steps the filter takes at each iteration
        - padding: The amount of zero padding to be added to the input structure
    '''
    self.input_shape = input_shape
    self.input_depth = input_shape[0]

    self.num_filters = filter_shape[0]
    self.filter_size = filter_shape[2]
    self.stride = stride
    self.padding = padding

    #self.output_shape = (depth, input_height - filter_size + 1, input_width - filter_size + 1)
    fan_in = input_shape[0] * filter_shape[2] * filter_shape[2]
    self.filter_shape = filter_shape
    self.filters = np.random.randn(filter_shape[0], input_shape[0], filter_shape[2], filter_shape[2])/np.sqrt(fan_in/2)
    self.biases = np.zeros(filter_shape[0])

  def zero_pad(self, input_data):
    '''
    This function pads the input with zeros to a certain degree.
    Input:
      - input_data: 2D array (H_1 x W_1)
    Output:
      - padded_input: 2D  padded array (H_1+padding X W_2+padding)
    '''

    batch_size, D_1, H_1, W_1 = input_data.shape
    #create a padded array of zeros
    padded_input = np.zeros((batch_size, D_1, H_1 + (2 * self.padding), W_1 + (2 * self.padding)))
    #copy the input structure into the centre of the padded array
    padded_input[:, :, self.padding:self.padding + H_1, self.padding:self.padding + W_1] = input_data

    return padded_input

  def stride_fun(self, input, h, w, filter_size):
    '''
    This function is meant to slide the filter along the structure volume a certain number of steps a
    at each iteration.
    Inputs:
      - h: The height of the structure
      - w: The width of the structure
      - filter_size: The width and/or the height of the filter
    Output:
      - The input slice at a given iteration
    '''
    stride = self.stride
     # Calculate the top-left corner of the current window
    h_start = h * stride
    w_start = w * stride

    F_w = filter_size
    F_h = filter_size
    # Extract and return the input slice
    return input[:, h_start:h_start + F_h, w_start:w_start + F_w]

  def forward(self, input_data):
    '''
    This function does the actual convolution process that we described ealier in this cookbook.
    '''
    self.input_data = input_data
    stride_num = self.stride
    padding = self.padding
    filter_size = self.filter_size
    filters = self.filters
    biases = self.biases

    (C_in, H_in, W_in) = self.input_shape
    batch_size = input_data.shape[0]
    (num_filters, C_in_filter, F_h, F_w) = self.filter_shape

    H_out = int(np.floor((H_in + 2*padding - F_h) / stride_num + 1))
    W_out = int(np.floor((W_in + 2*padding - F_w) / stride_num + 1))

    # Initialize the output tensor
    output = np.zeros((batch_size, num_filters, H_out, W_out))

    self.padded_input = self.zero_pad(input_data)

    # Perform convolution
    for b in range(batch_size):
      for n in range(num_filters):  # Loop over each filter
          for h in range(H_out):  # Loop over output height
              for w in range(W_out):  # Loop over output width
                  input_slice = self.stride_fun(self.padded_input[b], h, w, filter_size)

                  # Perform dot product
                  output[b, n, h, w] = np.sum(input_slice * filters[n]) + biases[n]

    return output

  def backward(self, output_gradient, optimizer):
    '''
    This function performs the backward pass to the network that will
    '''
    filters_gradient= np.zeros(self.filter_shape)
    input_gradient = np.zeros(self.padded_input.shape)
    bias_gradient = np.mean(output_gradient, axis=(0,2,3))

    #compute the gradients
    for b in range(self.input_data.shape[0]):
      for i in range (self.num_filters):
        for j in range (self. input_depth):
          filters_gradient[i, j] += signal.correlate2d(self.input_data[b,j], output_gradient[b,i], "valid")
          input_gradient[b,j] += signal.convolve2d(output_gradient[b,i], self.filters[i, j], "full")

    self.filters, self.biases = optimizer.update(
        self.filters, self.biases, filters_gradient, bias_gradient
    )

    return input_gradient