from Initializer import *
from Activation import *
from Loss import *
from Layer import *

class NeuralNetwork():
  """Deep neural network with multiple layers."""

  def __init__(self):
    self.layers = []
    self.loss = None
    self.costs = []
    self.learning_rate = None

  def add(self, layer):
    """Add layer to the network.

    Args: 
      layer: A layer to be added to the network
    """
    self.layers.append(layer)

  # Select the loss for model before training
  def compile(self, loss):
    """Configurations for training the network.
    
    Args:
      loss: Loss function
    """
    self.loss = Loss(loss)

  def fit(self, X, Y, epochs=1000, learning_rate=0.01, print_log=False):
    """Training the network.

    Args:
      X: Training example 
      Y: Ground truth label
      epochs: Number of loop through entire X
      learning_rate: Step size for updating parameters
      print_log: Print training log
    """
    self.learning_rate = learning_rate
    cost = 0

    for i in range(epochs):
      # Forward propagation
      output = X
      for layer in self.layers:
        output = layer.forward_propagation(output)
      cost = self.loss(Y, output)

      # Backward propagation
      error = self.loss(Y, output, derivative=True)
      for layer in reversed(self.layers):
        error = layer.backward_propagation(error, learning_rate)
      
      if (i%1000 == 0 or i == epochs - 1):
        self.costs.append(cost)

        if print_log: 
          print(f"Epochs: {i} - Cost: {cost}")
         
  def predict(self, X):
    """Predict output with given input."""
    input = X
    for layer in self.layers:
      input = layer.forward_propagation(input)
    return input

  def evaluate(self, X, Y):
    """Evaluation the network."""
    result = {"accuracy": (100 - np.mean(np.abs(Y - self.predict(X)))*100)}
    return result
