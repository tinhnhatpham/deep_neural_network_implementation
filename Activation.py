import numpy as np
from scipy.special import expit 
from scipy.special import xlogy
from scipy.special import xlog1py

class Activation:
  """Define activation functions and their derivative for deep neural network."""

  def __init__(self, name=None):
    self.name = name

  def __call__(self, x, derivative=False):
    if self.name is None:
      return None
    func_name = self.name if not derivative else self.name + "_prime"
    return getattr(self, func_name, None)(x)

  def sigmoid(self, x):
    return expit(x)
  
  def sigmoid_prime(self, x):
    return self.sigmoid(x)*(1 - self.sigmoid(x))

  def relu(self, x):
    return np.maximum(0, x)

  def relu_prime(self, x):
    return 1. * (x > 0)

  def tanh(self, x):
    return np.tanh(x);

  def tanh_prime(self, x):
    return 1-np.tanh(x)**2;    
