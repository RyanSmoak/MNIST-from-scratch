import numpy as np

class evaluation():
  def __init__(self, y_true, y_pred):
    '''
    Initialize the evaluation metrics.
    Input:
      y_true: The true class labels.
      y_pred: The predicted labels by our network.
    '''
    self.y_true = y_true
    self.y_pred = y_pred
  
  def accuracy(self):
    '''
    Compute the accuracy of the model.
    Output:
      The accuracy of the model.
    '''
    return np.mean(self.y_true == self.y_pred)

  def confusion_matrix(self, num_classes):
    '''
    Create a confusion matrix to show the relationship between classes.
    Output:
      The confusion matrix.
    '''
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(self.y_true, self.y_pred):
        conf_matrix[true_label, pred_label] += 1
    return conf_matrix    