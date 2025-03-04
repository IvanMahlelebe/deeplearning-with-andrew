import numpy as np
from typing import Tuple, Dict

from .activations import (
  sigmoid,
  relu,
  sigmoid_backward,
  relu_backward
)

class DeepNeuralNetwork:

  def __init__(self):
    self._seed = 42
    self._parameters: Dict[str, np.ndarray] = {}
    self._L: int = 0
    self._cache: Dict[str, Tuple[np.ndarray, ...]] = {}

  def initialize_params(self, layer_dims: Tuple[int, ...], seed: int = 42) -> None:
    """
      Initialises weights W with random values and biases b with 0s

      Args:
        layer_dims: Tuple[int, ...] - list of number of units per layer

      Returns:
        None
    """
    self._seed = seed
    np.random.seed(self._seed)
    if len(layer_dims) < 2:
      raise ValueError("layer_dims must have at least two elements")
    self._parameters.clear()  # Clear parameters dictionary
    self._L = len(layer_dims) - 1  # Number of layers

    for l in range(1, self._L + 1):
      self._parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
      self._parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

  def set_parameters(self, parameters):
    self._parameters = parameters

  def get_parameters(self) -> Dict[str, np.ndarray]:
    """
      Getter for the parameters of the neural network

      Returns:
        parameters: Dict[str, np.ndarray] - A dictionary containing the weights and biases.
    """
    return self._parameters

  def _linear_forward(
    self,
    A_prev: np.ndarray,
    W: np.ndarray,
    b: np.ndarray
  ) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """
    Computes the linear transformation Z = W * A_prev + b
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache
    
  def _activation_forward(
    self,
    Z: np.ndarray,
    activation: str
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies an activation function to Z.
    """
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError("Unsupported activation function")
    
    return A, activation_cache
  
  def _linear_activation_forward(
    self, 
    A_prev: np.ndarray, 
    W: np.ndarray, 
    b: np.ndarray, 
    activation: str
  ) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
    """
    Computes the forward pass for a single layer, including linear transformation and activation.
    """
    Z, linear_cache = self._linear_forward(A_prev, W, b)
    A, activation_cache = self._activation_forward(Z, activation)
    
    cache = (linear_cache, activation_cache)
    return A, cache
  
  def forward_propagation(
      self,
      X: np.ndarray,
      activations: Tuple[str, ...]
    ) -> np.ndarray:
    """
    Implements forward propagation for the entire network.
    """
    A = X
    self._cache = {}  # Reset cache
    
    for l in range(1, self._L):
      W = self._parameters[f"W{l}"]
      b = self._parameters[f"b{l}"]
      activation = activations[l-1] if l < self._L - 1 else "sigmoid"  # Allow flexible activations
      
      A, cache = self._linear_activation_forward(A, W, b, activation)
      self._cache[f"layer{l}"] = cache  # Store cache for backpropagation
    
    return A

  def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
    """
      Computes the cross-entropy cost.
    """
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    return np.squeeze(cost)
  
  def backward_propagation(
    self,
    AL: np.ndarray,
    Y: np.ndarray
  ) -> Dict[str, np.ndarray]:
    """
    Implements backpropagation for the entire network.
    """
    grads = {}
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    for l in reversed(range(1, self._L)):
      linear_cache, activation_cache = self._cache[f"layer{l}"]
      activation = "sigmoid" if l == self._L - 1 else "relu"
      
      if activation == "relu":
        dA_prev, dW, db = relu_backward(dAL, linear_cache, activation_cache)
      else:
        dA_prev, dW, db = sigmoid_backward(dAL, linear_cache, activation_cache)
      
      grads[f"dW{l}"] = dW
      grads[f"db{l}"] = db
      dAL = dA_prev
    
    return grads
  
  def predict(self, X: np.ndarray, activations: Tuple[str, ...]) -> np.ndarray:
    """
      Predicts the output for given input X.
    """
    AL = self.forward_propagation(X, activations)
    predictions = (AL > 0.5).astype(int)
    return predictions
  