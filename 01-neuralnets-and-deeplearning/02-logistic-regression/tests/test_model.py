import pytest
import numpy as np
from NeuralCatClassifier import NeuralCatIdentifier
import warnings

class TestNeuralCatIdentifier:

  @pytest.fixture(autouse=True)
  def suppress_warnings(self):
    warnings.simplefilter("ignore", category=DeprecationWarning)

  def test_initialization(self):
    dim = 2
    model = NeuralCatIdentifier(dim)
    w, b = model.get_initialised_params()
    
    expected_w = np.array([[0.], [0.]])
    expected_b = 0.0
    
    assert np.array_equal(w, expected_w), "Initialization of w failed"
    assert b == expected_b, "Initialization of b failed"

  def test_propagate(self):
    w = np.array([[1.], [2.]])
    b = 2.0
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    
    model = NeuralCatIdentifier(dim=2, w_init=w, b_init=b)
    grads, cost = model.propagate(X, Y)
    
    expected_dw = np.array([[0.99845601], [2.39507239]])
    expected_db = 0.00145557813678
    expected_cost = 5.801545319394553
    
    assert np.allclose(grads["dw"], expected_dw, rtol=1e-6), "Gradient dw is incorrect"
    assert np.isclose(grads["db"], expected_db, rtol=1e-6), "Gradient db is incorrect"
    assert np.isclose(cost, expected_cost, rtol=1e-6), "Cost is incorrect"

  def test_optimize(self):
    w = np.array([[1.], [2.]])
    b = 2.0
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    num_iterations = 100
    learning_rate = 0.009
    print_cost = False
    
    model = NeuralCatIdentifier(dim=2, w_init=w, b_init=b)
    params, grads, _ = model.optimize(X, Y, num_iterations, learning_rate, print_cost)
    
    expected_w = np.array([[0.19033591], [0.12259159]])
    expected_b = 1.92535983008
    expected_dw = np.array([[0.67752042], [1.41625495]])
    expected_db = 0.219194504541
    
    assert np.allclose(params["w"], expected_w, rtol=1e-6), "Optimized w is incorrect"
    assert np.isclose(params["b"], expected_b, rtol=1e-6), "Optimized b is incorrect"
    assert np.allclose(grads["dw"], expected_dw, rtol=1e-6), "Gradient dw is incorrect"
    assert np.isclose(grads["db"], expected_db, rtol=1e-6), "Gradient db is incorrect"

  def test_predict(self):
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    m = X.shape[1]
    
    model = NeuralCatIdentifier(dim=2, w_init=w, b_init=b)
    Y_prediction = model.predict(X)
    
    expected_Y_prediction = np.array([[1., 1., 0.]])
    
    assert Y_prediction.shape == (1, m), f"Expected shape (1, {m}), but got {Y_prediction.shape}"
    assert np.array_equal(Y_prediction, expected_Y_prediction), "Prediction is incorrect"

if __name__ == "__main__":
    pytest.main()
