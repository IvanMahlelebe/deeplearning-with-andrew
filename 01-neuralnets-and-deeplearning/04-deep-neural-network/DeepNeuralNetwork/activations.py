import numpy as np
from typing import Tuple

def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
      Computes the ReLU activation function.

      Args:
        Z (np.ndarray): Linear transformation output.

      Returns:
        A (np.ndarray): Activated output.
        cache (np.ndarray): Cached Z value for backpropagation.
    """
    A = np.maximum(0, Z)
    return A, Z  # Cache Z for backprop

def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
      Computes the sigmoid activation function.

      Args:
        Z (np.ndarray): Linear transformation output.

      Returns:
        A (np.ndarray): Activated output.
        cache (np.ndarray): Cached Z value for backpropagation.
    """
    A = 1 / (1 + np.exp(-Z))
    return A, Z  # Cache Z for backprop

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
      Computes the gradient of the ReLU activation.

      Args:
        dA (np.ndarray): Gradient of the activation output.
        cache (np.ndarray): Cached Z value from forward propagation.

      Returns:
        dZ (np.ndarray): Gradient of Z.
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # Copy dA to avoid modifying input
    dZ[Z <= 0] = 0  # Zero out gradients where Z was non-positive
    return dZ

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
  """
    Computes the gradient of the sigmoid activation.

    Args:
      dA (np.ndarray): Gradient of the activation output.
      cache (np.ndarray): Cached Z value from forward propagation.

    Returns:
      dZ (np.ndarray): Gradient of Z.
  """
  Z = cache
  s = 1 / (1 + np.exp(-Z))
  dZ = dA * s * (1 - s)  # Derivative of sigmoid function
  return dZ
