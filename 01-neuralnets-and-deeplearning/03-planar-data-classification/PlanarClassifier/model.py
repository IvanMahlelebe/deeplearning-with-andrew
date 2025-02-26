import numpy as np
from typing import Tuple, Optional

from PlanarClassifier.utilities import sigmoid

class PlanarClassifier:

  def __init__(self, input_layer_size: Optional[int] = None, output_layer_size: Optional[int] = None, hidden_layer_size: Optional[int] = 4):
    """
      Initializes a neural network with optional layer sizes.
      Parameters are initialized only when all layer sizes are provided.
    """
    self._input_layer_size = input_layer_size
    self._hidden_layer_size = hidden_layer_size
    self._output_layer_size = output_layer_size
    self._parameters_initialized = False
    self._parameters = {}

  def set_layer_sizes(self, X, Y):
    self._input_layer_size = X.shape[0]
    self._output_layer_size = Y.shape[0]

  def get_layer_sizes(self) -> Tuple[int, int, int]:
    """returns input, hidden and output layer sizes"""
    return (
      self._input_layer_size,
      self._hidden_layer_size,
      self._output_layer_size,
    )
  
  def initialize_parameters(self) -> None:
    """
      Initializes the model parameters (weights and biases) based on the layer sizes.
      Raises an error if layer sizes are not set.
    """
    np.random.seed(2)
    if self._input_layer_size is None or self._hidden_layer_size is None or self._output_layer_size is None:
      raise ValueError("Layer sizes must be set before initializing parameters.")

    self._parameters['W1'] = np.random.randn(self._hidden_layer_size, self._input_layer_size) * 0.01
    self._parameters['b1'] = np.zeros((self._hidden_layer_size, 1))
    self._parameters['W2'] = np.random.randn(self._output_layer_size, self._hidden_layer_size) * 0.01
    self._parameters['b2'] = np.zeros((self._output_layer_size, 1))

    self._parameters_initialized = True

  def set_parameters(self, parameters: dict) -> None:
    """
      Sets the model parameters.

      Args:
        parameters (dict): A dictionary containing the model parameters with keys:
          - 'W1': Weight matrix for the first layer (shape: hidden_layer_size x input_layer_size)
          - 'b1': Bias vector for the first layer (shape: hidden_layer_size x 1)
          - 'W2': Weight matrix for the second layer (shape: output_layer_size x hidden_layer_size)
          - 'b2': Bias vector for the second layer (shape: output_layer_size x 1)

      Raises:
        ValueError: If parameters have already been initialized, or if the input dictionary is invalid.
    """
    if self._parameters_initialized:
      raise ValueError("Parameters have already been initialized.")

    required_keys = {'W1', 'b1', 'W2', 'b2'}
    if not isinstance(parameters, dict) or not required_keys.issubset(parameters.keys()):
      raise ValueError(f"Parameters must be a dictionary with keys: {required_keys}.")

    self._parameters = parameters
    self._parameters_initialized = True

  def get_parameters(self) -> dict:
    """
      Returns the model parameters.
      Raises an error if parameters are not initialized.
    """
    if not self._parameters_initialized:
      raise ValueError("Parameters are not initialized. Call `initialize_parameters()` first.")
    return self._parameters

  def forward_propagation(self, X):
    """
      Arguments:
      - X: input data of size (input_size, num_examples)

      Returns:
      - A2: sigmoid output of the second derivative
      - cache: dict(Z1, A1, Z2, A2)
    """

    parameters = self.get_parameters()
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    cache = {
      "Z1": Z1,
      "A1": A1,
      "Z2": Z2,
      "A2": A2
    }

    return A2, cache
  
  def get_cost(self, A2, Y):
    """
      Compute the cross-entropy loss
    """
    
    EXAMPLES_NUM = Y.shape[1]

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (-1/EXAMPLES_NUM) * np.sum(logprobs)
    cost = float(np.squeeze(cost))

    return cost

  def backward_propagation(self, cache, X, Y):
    m = X.shape[1]

    parameters = self.get_parameters()
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) *(np.sum(dZ2,axis=1,keepdims=True))
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
    dW1 = (1/m) *(np.dot(dZ1,X.T))
    db1 = (1/m) *(np.sum(dZ1, axis=1, keepdims=True))

    grads = {
      "dW1": dW1,
      "db1": db1,
      "dW2": dW2,
      "db2": db2,
    }

    return grads

  def update_parameters(self, grads, learning_rate):
    
    parameters = self.get_parameters()
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    updated_params = {
      "W1": W1,
      "b1": b1,
      "W2": W2,
      "b2": b2
    }

    return updated_params

  def train(self, X, Y, learning_rate, num_iterations = 10000) -> Tuple[dict, dict]:
    
    cost_updates = {}
    for iteration in range(0, num_iterations):
      A2, cache = self.forward_propagation(X)
      cost = self.get_cost(A2, Y)
      grads = self.backward_propagation(cache, X, Y)
      updated_parameters = self.update_parameters(grads, learning_rate)
      cost_updates[iteration] = cost

    return updated_parameters, cost_updates

  def predict(self, X):
    if not self._parameters_initialized:
      raise ValueError("Parameters not initialized.")
    A2, _ = self.forward_propagation(X)
    predictions = (A2 > 0.5)
    return predictions