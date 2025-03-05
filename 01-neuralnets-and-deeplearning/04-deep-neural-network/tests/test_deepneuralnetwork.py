import numpy as np

from DeepNeuralNetwork.DeepNeuralNetwork import DeepNeuralNetwork

class TestDeepNeuralNetwork:
  def test_initialize_params(self):
    SEED = 3
    np.random.seed(SEED)
    layer_dims = (5, 4, 3)
    dnn = DeepNeuralNetwork(seed=SEED)
    dnn.initialize_params(layer_dims)
    model_parameters = dnn.get_parameters()
    
    expected_shapes = {
        "W1": (4, 5),
        "b1": (4, 1),
        "W2": (3, 4),
        "b2": (3, 1),
    }
    expected_parameters = {
      'W1': np.array([
        [ 0.01788628,  0.0043651,   0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
        [-0.00404677, -0.0054536,  -0.01546477,  0.00982367, -0.01101068]
      ]),
      'b1': np.array([[0.], [0.], [0.], [0.]]),
      'W2': np.array([
        [-0.01185047, -0.0020565,   0.01486148,  0.00236716],
        [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
        [-0.00768836, -0.00230031,  0.00745056,  0.01976111]
      ]),
      'b2': np.array([[0.], [0.], [0.]])
    }

    for l in range(1, len(layer_dims)):
      assert model_parameters[f"W{l}"].shape == expected_shapes[f"W{l}"]
      assert model_parameters[f"b{l}"].shape == expected_shapes[f"b{l}"]
      np.testing.assert_allclose(
        model_parameters[f"W{l}"],
        expected_parameters[f"W{l}"],
        atol=1e-8,
        err_msg=f'Incorrect. Expected {expected_parameters[f"W{l}"]}; Got {model_parameters[f"W{l}"]}'
      )
      np.testing.assert_allclose(
        model_parameters[f"b{l}"],
        expected_parameters[f"b{l}"],
        atol=1e-8,
        err_msg=f'Incorrect. Expected {expected_parameters[f"W{l}"]}; Got {model_parameters[f"W{l}"]}'
      )

  def test_linear_forward(self):
    SEED = 1
    np.random.seed(SEED)
    dnn = DeepNeuralNetwork(seed=SEED)
    dnn.initialize_params((3, 2, 1))

    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    expected_z = np.array([[ 3.26295337, -1.23429987]])
    Z, _ = dnn._linear_forward(A_prev, W, b)

    assert (Z.shape == (W.shape[0], A_prev.shape[1])), f"Incorrect. Expected {Z.shape}; Got {W.shape[0], A_prev.shape[1]}"
    # np.testing.assert_allclose(
    #   Z,
    #   expected_z,
    #   atol=1e-8,
    #   err_msg=f'Incorrect. Expected {expected_z}; Got {Z}'
    # )

  def test_linear_activation_forward(self):
    SEED = 2
    np.random.seed(SEED)
    dnn = DeepNeuralNetwork(seed=SEED)
    dnn.initialize_params((3, 2, 1))

    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    A1, _ = dnn._linear_activation_forward(A_prev, W, b, activation="sigmoid")
    A2, _ = dnn._linear_activation_forward(A_prev, W, b, activation="relu")

    expected_A1 = np.array([[0.96890023, 0.11013289]])
    expected_A2 = np.array([[3.43896131, 0.]])

    assert (A1.shape == (W.shape[0], A_prev.shape[1])), f"Incorrect. Expected {A1.shape}; Got {W.shape[0], A_prev.shape[1]}"
    assert (A2.shape == (W.shape[0], A_prev.shape[1])), f"Incorrect. Expected {A2.shape}; Got {W.shape[0], A_prev.shape[1]}"
    # np.testing.assert_allclose(
    #   A1,
    #   expected_A1,
    #   atol=1e-8,
    #   err_msg=f'Incorrect. Expected {expected_A1}; Got {A1}'
    # )
    # np.testing.assert_allclose(
    #   A2,
    #   expected_A2,
    #   atol=1e-8,
    #   err_msg=f'Incorrect. Expected {expected_A2}; Got {A2}'
    # )

  def test_forward_propagation(self):
    SEED = 1
    np.random.seed(SEED)
    dnn = DeepNeuralNetwork(seed=SEED)

    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {
      "W1": W1,
      "b1": b1,
      "W2": W2,
      "b2": b2
    }
    activations = ["relu", "relu", "relu", "sigmoid"]

    dnn.set_parameters(parameters)
    A = dnn.forward_propagation(X, activations)

    assert (A.shape == (1,X.shape[1])), f"Incorrect. Expected {A.shape}; Got {1,X.shape[1]}"