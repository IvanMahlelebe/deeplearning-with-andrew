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
    self._seed = 0
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
    linear_cache = (A_prev, W, b)
    return Z, linear_cache
    
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
      activation = activations[l]
      
      A, cache = self._linear_activation_forward(A, W, b, activation)
      self._cache[f"layer{l}"] = cache
    
    return A

  def compute_cost(self, Y: np.ndarray) -> float:
    """
      Computes the cross-entropy cost.
    """
    m = Y.shape[1]
    AL = self._cache[f"layer{self._L}"]
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot((1 - Y), (np.log(1 - AL).T)))
    return np.squeeze(cost)

  def _linear_backward(
    self,
    dZ: np.ndarray,
    linear_cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the gradients for the linear transformation.
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db
  
  def _activation_backward(
    self,
    dA: np.ndarray,
    activation_cache: np.ndarray,
    activation: str
  ) -> np.ndarray:
    """
    Computes the gradient of the cost with respect to Z.
    """
    Z = activation_cache

    if activation == "sigmoid":
      dZ = dA * sigmoid_backward(Z)
    elif activation == "relu":
      dZ = dA * relu_backward(Z)
    else:
      raise ValueError("Unsupported activation function")

    return dZ
  
  def _linear_activation_backward(
    self,
    dA: np.ndarray,
    cache: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    activation: str
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the backward pass for a single layer, including linear transformation and activation.
    """
    linear_cache, activation_cache = cache

    dZ = self._activation_backward(dA, activation_cache, activation)
    dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

  def backward_propagation(
    self,
    Y: np.ndarray,
    activations: Tuple[str, ...]
  ) -> Dict[str, np.ndarray]:
    """
    Implements backward propagation for the entire network.
    """
    grads = {}
    m = Y.shape[1]
    L = self._L

    AL = self._cache[f"layer{L}"]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = self._cache[f"layer{L}"]
    grads[f"dA{L}"], grads[f"dW{L}"], grads[f"db{L}"] = self._linear_activation_backward(
        dAL, current_cache, activations[L]
    )

    for l in reversed(range(1, L)):
      current_cache = self._cache[f"layer{l}"]
      dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(
        grads[f"dA{l+1}"], current_cache, activations[l]
      )
      grads[f"dA{l}"] = dA_prev_temp
      grads[f"dW{l}"] = dW_temp
      grads[f"db{l}"] = db_temp

    return grads
  
  # def backward_propagation(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
  #   """
  #   Implements backpropagation for the entire network.
  #   """
  #   grads = {}
  #   m = Y.shape[1]
  #   AL = self._cache[f"layer{self._L}"]
  #   Y = Y.reshape(AL.shape)
  #   dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
  #   for l in reversed(range(1, self._L)):
  #     linear_cache, activation_cache = self._cache[f"layer{l}"]
  #     activation = "sigmoid" if l == self._L - 1 else "relu"
      
  #     if activation == "relu":
  #       dA_prev, dW, db = relu_backward(dAL, linear_cache, activation_cache)
  #     else:
  #       dA_prev, dW, db = sigmoid_backward(dAL, linear_cache, activation_cache)
      
  #     grads[f"dW{l}"] = dW
  #     grads[f"db{l}"] = db
  #     dAL = dA_prev
    
  #   return grads
  
  # def predict(self, X: np.ndarray, activations: Tuple[str, ...]) -> np.ndarray:
  #   """
  #     Predicts the output for given input X.
  #   """
  #   AL = self.forward_propagation(X, activations)
  #   predictions = (AL > 0.5).astype(int)
  #   return predictions
  