import pytest
import numpy as np
from NeuralCatClassifier import DataPreprocessor

class TestDataPreprocessor:

  def test_flatten_data_train(self):
    train_set_x_orig = np.random.rand(209, 64, 64, 3)
    preprocessor = DataPreprocessor()
    flattened_data = preprocessor.flatten_data(train_set_x_orig)
    expected_shape = (12288, 209)
    assert flattened_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {flattened_data.shape}"

  def test_flatten_data_test(self):
    test_set_x_orig = np.random.rand(50, 64, 64, 3)
    preprocessor = DataPreprocessor()
    flattened_data = preprocessor.flatten_data(test_set_x_orig)
    expected_shape = (12288, 50)
    assert flattened_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {flattened_data.shape}"


if __name__ == "__main__":
  pytest.main(["-v", "-W", "ignore::DeprecationWarning"])