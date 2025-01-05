import numpy as np

class BatchNormalization:
  def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
    '''
    I really don't know what else I can say about initilizations.
    Input:
      input_size: batch size
      epsilon: a small positive number to avoid division by 0
      momentum: momentum coefficient (friction) when using SGD with momentum
    '''
    self.gamma = np.ones((input_size, 1)) 
    self.beta = np.zeros((input_size, 1)) 
    self.epsilon = epsilon
    self.momentum = momentum
    self.running_mean = np.zeros((input_size, 1))
    self.running_var = np.zeros((input_size, 1))

  def forward(self, x, training=True):
    '''
    This is the forward pass for batch normalization
    Input: 
      training: a boolean telling us whether the network should learn gamma and beta or not
    Output:
      The normalized output 
    '''
    if training:
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        
        # Update running estimates
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
    else:
        # Use running estimates for inference
        self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
    
    self.out = self.gamma * self.x_hat + self.beta
    return self.out

  def backward(self, dout):
    '''
    This function is meant to 'undo' the effect of the normalization by backprop
    so as to compute the gradient of x
    Input:
      dout: the gradient of a later layer in the network

    Output:
      dx: the gradient of x backpropagated through the batch normalization 
    '''
    m = dout.shape[1]
    
    # Gradients w.r.t. gamma and beta
    self.dgamma = np.sum(dout * self.x_hat, axis=1, keepdims=True)
    self.dbeta = np.sum(dout, axis=1, keepdims=True)
    
    # Backprop through normalization
    dx_hat = dout * self.gamma
    dvar = np.sum(dx_hat * (self.input - self.mean) * -0.5 * (self.var + self.epsilon)**-1.5, axis=1, keepdims=True)
    dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.epsilon), axis=1, keepdims=True) + dvar * np.mean(-2 * (self.input - self.mean), axis=1, keepdims=True)
    
    dx = dx_hat / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.input - self.mean) / m + dmean / m
    return dx
