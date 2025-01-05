class Flatten:
  def __init__(self):
    '''
    An initialization function for the flatten layer.
    I think by now you should know what I'm going to say for this
    '''
    self.input_shape = None #to store the original shape for unflatening 

  def flatten(self, input_tensor):
    '''
    Reshape the input tensor into a 1D vector.
    Input:
      input_tensor: The input tensor to be flattened.
    Output:
      The flattened tensor.
    '''
    self.input_shape = input_tensor.shape
    return input_tensor.reshape(-1)

  def unflatten(self, output_gradient):
    '''
    Reshape the output gradient from the FC layer into original shape
    Input:
      output_gradient: The output gradient from the FC layer.
    Output:
      The unflattened gradient.
    '''
    return output_gradient.reshape(self.input_shape)