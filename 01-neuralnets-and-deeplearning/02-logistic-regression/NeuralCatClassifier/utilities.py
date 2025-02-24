import h5py
import numpy as np

def load_dataset():
  try:
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
      train_set_x_orig = np.array(train_dataset["train_set_x"][:])
      train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
      test_set_x_orig = np.array(test_dataset["test_set_x"][:])
      test_set_y_orig = np.array(test_dataset["test_set_y"][:])
      classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

  except FileNotFoundError as e:
    print(f"Error: {e}. Please check that the dataset files exist.")
  except OSError as e:
    print(f"Error: {e}. The file might be corrupted or not an HDF5 file.")
  except Exception as e:
    print(f"Unexpected error: {e}")

  return None

def sigmoid(z):
  """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
  """

  s = 1 / (1 + np.exp(-z))
  
  return s