import pytest
import numpy as np
from PlanarClassifier.utilities import (
  create_flower_data
)


class TestUtilities:

  def test_create_flower_data(self):
    """
      Test the create_flower_data function to ensure it returns the correct shapes and number of examples.
    """
    
    X, Y = create_flower_data()
    
    assert X.shape == (2, 400), f"Expected shape of X to be (2, 400), but got {X.shape}"
    assert Y.shape == (1, 400), f"Expected shape of Y to be (1, 400), but got {Y.shape}"
    assert X.shape[1] == 400, f"Expected number of training examples to be 400, but got {X.shape[1]}"
    assert Y.shape[1] == 400, f"Expected number of training examples to be 400, but got {Y.shape[1]}"
    assert np.all(np.logical_or(Y == 0, Y == 1)), "Y should only contain values 0 or 1"
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(Y, np.ndarray), "Y should be a numpy array"
    assert not np.all(X == 0), "X should not be all zeros"
    assert not np.all(Y == 0), "Y should not be all zeros"

if __name__ == "__main__":
  pytest.main()