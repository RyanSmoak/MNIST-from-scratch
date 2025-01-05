import numpy as np

class losses():
  def __init__(self, y_true, y_pred):
    '''
    Initialize the loss function.
    Input:
      y_true: The true labels.
      y_pred: The predicted labels.
    '''
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

    # Ensure y_true is an integer array of class indices
    y_true = self.y_true.astype(int)

    if y_true.ndim > 1:
      y_true = np.argmax(y_true, axis=1)

    correct_class_probs = self.y_pred[np.arange(batch_size), y_true]
    log_probs = np.log(correct_class_probs + 1e-15)

    log_probs = log_probs[:, np.newaxis]

    return -np.mean(np.sum(self.y_true * log_probs, axis =1))