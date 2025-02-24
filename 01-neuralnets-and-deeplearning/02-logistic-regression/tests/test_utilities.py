import pytest
import numpy as np
import h5py
import os
from NeuralCatClassifier.utilities import (
  load_dataset,
  sigmoid
)

@pytest.fixture
def mock_dataset_files(tmpdir):
  train_file = tmpdir.join("train_catvnoncat.h5")
  test_file = tmpdir.join("test_catvnoncat.h5")

  train_set_x = np.random.rand(209, 64, 64, 3)
  train_set_y = np.random.randint(0, 2, size=(209,))
  test_set_x = np.random.rand(50, 64, 64, 3)
  test_set_y = np.random.randint(0, 2, size=(50,))
  classes = np.array([b"non-cat", b"cat"])

  with h5py.File(train_file, "w") as f:
    f.create_dataset("train_set_x", data=train_set_x)
    f.create_dataset("train_set_y", data=train_set_y)

  with h5py.File(test_file, "w") as f:
    f.create_dataset("test_set_x", data=test_set_x)
    f.create_dataset("test_set_y", data=test_set_y)
    f.create_dataset("list_classes", data=classes)

  return str(train_file), str(test_file)

class TestLoadDataset:

  def test_load_dataset_success(self, mock_dataset_files):
    train_file, test_file = mock_dataset_files

    dataset = load_dataset()

    assert dataset is not None, "Dataset loading failed"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = dataset

    assert train_set_x_orig.shape == (209, 64, 64, 3), "Train set x shape is incorrect"
    assert train_set_y.shape == (1, 209), "Train set y shape is incorrect"
    assert test_set_x_orig.shape == (50, 64, 64, 3), "Test set x shape is incorrect"
    assert test_set_y.shape == (1, 50), "Test set y shape is incorrect"
    assert classes.shape == (2,), "Classes shape is incorrect"

  def test_load_dataset_file_not_found(self, tmpdir):
    os.chdir(tmpdir)  # Change to a directory without the dataset files
    dataset = load_dataset()
    assert dataset is None, "Expected None when dataset files are missing"

  def test_load_dataset_corrupted_file(self, mock_dataset_files):
    train_file, test_file = mock_dataset_files
    with open(train_file, "w") as f:
      f.write("corrupted data")

    dataset = load_dataset()

    assert dataset is None, "Expected None when dataset file is corrupted"

class TestSigmoid:

  def test_sigmoid(self):
    z = np.array([0, 2])
    expected_output = np.array([0.5, 0.88079708])
    output = sigmoid(z)
    assert np.allclose(output, expected_output, rtol=1e-6), "Sigmoid output is incorrect"

if __name__ == "__main__":
    pytest.main(["-v", "-W", "ignore::DeprecationWarning"])