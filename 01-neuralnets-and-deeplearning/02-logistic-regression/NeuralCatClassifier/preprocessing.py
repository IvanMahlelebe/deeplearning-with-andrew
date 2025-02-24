import numpy as np

class DataPreprocessor:

  def flatten_data(self, data):
    flattened_data = (
      data
      .reshape(
        np.power(data.shape[1], 2)*3,
        -1
      )
    )

    return flattened_data