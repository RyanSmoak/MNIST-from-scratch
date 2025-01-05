import numpy as np

class loss():
  def __init__(self, y_true, y_pred):
    '''
    Initialize the loss function.
    Input:
      y_true: The true labels.
      y_pred: The predicted labels.
    '''
    self.m = y_true.shape[0]
    self.y_true = y_true
    self.y_pred = y_pred

  def binary_cross_entropy(self):
    '''
    Compute the binary cross-entropy loss.
    Output:
      The binary cross-entropy loss.
    '''
    return -(1/self.m) * np.sum(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))


  def cross_entropy(self):
    '''
    Compute the cross-entropy loss for multi-class classification.
    Output:
      The cross-entropy loss.
    '''    
    batch_size = self.y_pred.shape[0] 
    correct_class_probs = self.y_pred[np.arange(batch_size), self.y_true]
    log_probs = np.log(correct_class_probs + 1e-15)
    return -np.mean(log_probs)