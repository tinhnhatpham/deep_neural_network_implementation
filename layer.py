from initalizer import *
from activation import *


class Layer:
  """Define abtract base layer class."""

  def __init__(self):
    self.input = None
    self.output = None

  # Perform forward propagation for current layer
  def forward_propagation(self, input):
    raise NotImplementedError

  # Perform backward propagation for current layer
  def backward_propagation(self, output, learning_rate=0.01):
    raise NotImplementedError
    
class LinearLayer(Layer):
  """A layer that performs a linear transformation on its input."""

  def __init__(self, units, initializer='random_normal'):
    self.units = units

    # Check initializer instance
    if isinstance(initializer, Initializer):
      self.initializer = initializer
    else:
      self.initializer = Initializer(initializer)

    self.is_initialized = False

    self.weights = None
    self.bias = None

  def forward_propagation(self, input):
    if not self.is_initialized:
      self.weights, self.bias = self.initializer((self.units, input.shape[0]))
      self.is_initialized = True
      
    self.input = input
    # Linear transformation
    self.output = np.dot(self.weights, input) + self.bias
    return self.output

  def backward_propagation(self, output, learning_rate=0.01):
    m = self.input.shape[1]
    d_weights = 1/m * np.dot(output, self.input.T)
    d_bias = 1/m * np.sum(output, axis=1, keepdims=True)

    # Update parameters
    self.weights -= learning_rate * d_weights
    self.bias -= learning_rate * d_bias

    return np.dot(self.weights.T, output)
  
class ActivationLayer(Layer):
  """A layer that applies a non-linear transformation to the input."""
  
  def __init__(self, activation_name=None):
    self.activation = Activation(activation_name)

  def forward_propagation(self, input):
    self.input = input
    self.output = self.activation(input)
    return self.output

  def backward_propagation(self, output):
    return output * self.activation(self.input, derivative=True)
  
class Dense(Layer):
  """A layer that combines linear and non-linear transformation of its input."""
  
  def __init__(self, units, activation, initializer='random_normal'):
    self.linearLayer = LinearLayer(units, initializer)
    self.activationLayer = ActivationLayer(activation)
    
  def forward_propagation(self, input):
    self.input = input
    self.output = self.linearLayer.forward_propagation(input)
    self.output = self.activationLayer.forward_propagation(self.output)
    return self.output

  def backward_propagation(self, output, learning_rate=0.01):
    self.output = self.activationLayer.backward_propagation(output)
    self.output = self.linearLayer.backward_propagation(self.output, learning_rate)
    return self.output
  
