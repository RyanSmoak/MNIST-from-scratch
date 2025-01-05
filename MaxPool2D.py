import numpy as np

class MaxPool2D():
  def __init__(self, pool_size, stride, padding):
    self.pool_size = pool_size
    self.stride = stride
    self.padding = padding
    self.input_shape = None
    self.input_map = None

  def forward(self, feature_map):
    '''
    This function is meant to act as the pooling layer after a Conv2D layer
    Inputs:
      feature_map: This is the output of the Conv2D layer
      pool_size: This is the size of the pooling filter
      stride: This the steps to be taken by the filter
      padding: This is the amount of zero padding to be added
    Output:
      output: This is the input structure with reduced spatial dimanesions of the pooling layer
    '''
    #account for any padding that may be added

    self.input_shape = feature_map.shape
    self.input_map = feature_map

    if self.padding > 0:
          feature_map = np.pad(feature_map,
                        ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
                        mode='constant',
                        constant_values=0)

    #Get the shapes for the input and the pooling filter
    batch_size, channels, H_in, W_in = feature_map.shape
    pool_h, pool_w = self.pool_size

    #calculate the output size
    W_out = (W_in - pool_w) // self.stride + 1
    H_out = (H_in - pool_h) // self.stride + 1

    H_out = max(1, H_out)
    W_out = max(1, W_out)

    self.max_indices = np.zeros_like(feature_map)

    #initialize the output
    output_map = np.zeros((batch_size, channels, H_out, W_out))

    #create the window and sliding
    for b in range(batch_size):
      for c in range(channels):
        for i in range(H_out):
          for j in range(W_out):
            #slide the window
            window = feature_map[b, c, i*self.stride : i*self.stride+pool_h,
                                j*self.stride : j*self.stride+pool_w]
            #perform max pooling
            output_map[b, c, i, j] = np.max(window)
            max_idx = np.unravel_index(np.argmax(window), window.shape)
            self.max_indices[b, c, i+max_idx[0], j+max_idx[1]] = 1


    return output_map

  def backward(self, dL_dOutput):
    '''
    This function is meant to propagate the feature map with repect to the input.
    Basically give us our gradient the same shape as the input
    Input:
      dL_dOutput: Gradients passed in the backprop
    '''
    dL_dInput = np.zeros_like(self.input_map)
    batch_size, channels, out_height, out_width = dL_dOutput.shape

    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_height):
                for j in range(out_width):
                    # Find the corresponding window in the input
                    window_start_i = i * self.stride
                    window_start_j = j * self.stride
                    window_end_i = window_start_i + self.pool_size[0]
                    window_end_j = window_start_j + self.pool_size[0]

                    # Only the position of the max value gets the gradient
                    max_mask = self.max_indices[b, c, window_start_i:window_end_i, window_start_j:window_end_j]
                    dL_dInput[b, c, window_start_i:window_end_i, window_start_j:window_end_j] += dL_dOutput[b, c, i, j] * max_mask

    return dL_dInput