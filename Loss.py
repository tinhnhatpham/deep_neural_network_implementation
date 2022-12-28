import numpy as np

class Loss():
  """Compute the cost of the predicted output of the model and the true output."""
  
  def __init__(self, name):
    self.name = name
  
  def __call__(self, Y, Y_pred, derivative=False):
    func_name = self.name if not derivative else self.name + "_prime"
    return getattr(self, func_name)(Y, Y_pred)

  def crossentropy(self, Y, Y_pred):
    m = Y.shape[1]
    # Add a very small epsilon to avoid log(0)
    epsilon = 1e-5
    cost = np.multiply(Y, np.log(Y_pred + epsilon)) + np.multiply(1 - Y, np.log(1 - Y_pred + epsilon))
    return np.squeeze(-1/m * np.sum(cost))

  def crossentropy_prime(self, Y, Y_pred):
    epsilon = 1e-5
    return - (np.divide(Y, Y_pred + epsilon) - np.divide(1 - Y, 1 - Y_pred + epsilon))
