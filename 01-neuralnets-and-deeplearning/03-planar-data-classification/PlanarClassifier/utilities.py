import numpy as np
import plotly.graph_objects as go
from typing import Tuple


def create_flower_data() -> Tuple[np.ndarray, np.ndarray]:
  """
    Generates a 2D flower-shaped dataset with two classes.

    Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - X (np.ndarray): A 2D array of shape (2, 400) representing the feature matrix.
        - Y (np.ndarray): A 2D array of shape (1, 400) representing the labels.
  """
  
  SEED = 42
  np.random.seed(SEED)

  NUM_EXAMPLES = 400
  DIMENSIONALITY = NUM_CLASSES = 2
  X = np.zeros((NUM_EXAMPLES, DIMENSIONALITY))
  Y = np.zeros((NUM_EXAMPLES, 1), dtype='uint8')
  MAX_RADIUS = 4

  points_per_class = NUM_EXAMPLES // NUM_CLASSES
  for j in range(2):
    ix = range(points_per_class * j, points_per_class * (j + 1))
    theta = np.linspace(j * 3.12, (j + 1) * 3.12, points_per_class) + np.random.randn(points_per_class) * 0.2
    radius = MAX_RADIUS * np.sin(4 * theta) + np.random.randn(points_per_class) * 0.2
    X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
    Y[ix] = j

  X = X.T
  Y = Y.T

  return X, Y


def plot_decision_boundary(model, X, y):
  """
    Plots the decision boundary of a model along with the data points using Plotly.

    Args:
      model (callable): A trained model that can predict labels for input data.
      X (np.ndarray): Input data of shape (2, m), where m is the number of examples.
      y (np.ndarray): True labels of shape (1, m).
  """
  
  x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
  y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
  h = 0.01

  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  Z = model(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  contour = go.Contour(
    x=np.arange(x_min, x_max, h),
    y=np.arange(y_min, y_max, h),
    z=Z,
    colorscale=['#FBF5DD','#A6CDC6'],
    opacity=0.5,
    name='Decision Boundary'
  )

  scatter = go.Scatter(
    x=X[0, :],
    y=X[1, :],
    mode='markers',
    marker=dict(
      color=y.ravel(),
      colorscale=['#DDA853','#16404D'],
      line=dict(color='black', width=1)
    ),
    name='Data Points'
  )

  fig = go.Figure(data=[contour, scatter])

  fig.update_layout(
    title='Model Decision Boundary',
    xaxis_title='x1',
    yaxis_title='x2',
    width=800,
  )

  fig.show()

def sigmoid(x):
  """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
  """

  s = 1 / (1 + np.exp(-x))
  
  return s