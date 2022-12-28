import numpy as np

class Initializer():
  def __init__(self, method="random_normal", scale=1):
    self.method = method
    self.shape = None
    self.scale = scale

  def __call__(self, shape):
    self.shape = shape
    func_name = self.method
    return getattr(self, func_name)()

  def random_normal(self):
    weights = np.random.randn(*self.shape) * self.scale
    bias = np.zeros((self.shape[0], 1)) * self.scale
    return weights, bias
  
  def he(self):
    scale = np.sqrt(2/self.shape[1])
    weights = np.random.randn(*self.shape) * scale
    bias = np.zeros((self.shape[0], 1)) * scale
    return weights, bias

  def xavier(self):
    scale = np.sqrt(1/self.shape[1])
    weights = np.random.randn(*self.shape) * scale
    bias = np.zeros((self.shape[0], 1)) * scale
    return weights, bias
