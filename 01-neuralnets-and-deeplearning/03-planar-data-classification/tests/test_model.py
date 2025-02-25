import numpy as np

from PlanarClassifier.model import PlanarClassifier

class TestModel:

  def test_set_layer_sizes(self):
    np.random.seed(1)
    model = PlanarClassifier()
    X = np.random.randn(5, 3)
    Y = np.random.randn(2, 3)
    model.set_layer_sizes(X, Y)
    model_layer_sizes = model.get_layer_sizes()
    expected_layer_sizes = (5, 4, 2)
    assert (model_layer_sizes == expected_layer_sizes), f"Incorrect layer sizes. Expected {expected_layer_sizes}, instead got {model_layer_sizes}"

  def test_initialize_parameters(self):
    """
      Test that parameters are initialized correctly when all layer sizes are set.
    """
    model = PlanarClassifier(input_layer_size=2, hidden_layer_size=4, output_layer_size=1)
    model.initialize_parameters()
    parameters = model.get_parameters()

    expected_W1 = np.array([
      [-0.00416758, -0.00056267],
      [-0.02136196, 0.01640271],
      [-0.01793436, -0.00841747],
      [0.00502881, -0.01245288]
    ])
    expected_b1 = np.array([[0.], [0.], [0.], [0.]])
    expected_W2 = np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]])
    expected_b2 = np.array([[0.]])

    np.testing.assert_allclose(parameters['W1'], expected_W1, atol=1e-8)
    np.testing.assert_allclose(parameters['b1'], expected_b1, atol=1e-8)
    np.testing.assert_allclose(parameters['W2'], expected_W2, atol=1e-8)
    np.testing.assert_allclose(parameters['b2'], expected_b2, atol=1e-8)

  def test_forward_propagation(self):
    """
      Test the forward_propagation method of the PlanarClassifier class.
    """
    np.random.seed(1)

    X = np.random.randn(2, 3)

    b1 = np.random.randn(4, 1)
    b2 = np.array([[-1.3]])
    parameters = {
      'W1': np.array(
        [
          [-0.00416758, -0.00056267],
          [-0.02136196, 0.01640271],
          [-0.01793436, -0.00841747],
          [0.00502881, -0.01245288]
        ]
      ),
      'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
      'b1': b1,
      'b2': b2
    }

    model = PlanarClassifier()
    model.set_parameters(parameters)

    A2, cache = model.forward_propagation(X)

    output_means = (
      np.mean(cache['Z1']),
      np.mean(cache['A1']),
      np.mean(cache['Z2']),
      np.mean(cache['A2'])
    )

    expected_means = (
      0.26281864019752443,
      0.09199904522700109,
      -1.3076660128732143,
      0.21287768171914198
    )

    np.testing.assert_allclose(
      output_means,
      expected_means,
      rtol=1e-8,
      atol=1e-8,
      err_msg=f"Incorrect. Expected {expected_means}; Got {output_means}"
    )

  def test_get_cost(self):
    np.random.seed(1)
    Y = (np.random.randn(1, 3) > 0)
    parameters = {
      'W1': np.array(
        [
          [-0.00416758, -0.00056267],
          [-0.02136196,  0.01640271],
          [-0.01793436, -0.00841747],
          [ 0.00502881, -0.01245288]
        ]
      ),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[0.], [0.], [ 0.], [0.]]),
     'b2': np.array([[0.]])
    }
    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))

    model = PlanarClassifier()
    model.set_parameters(parameters)

    expected_cost = 0.6930587610394646
    cost = model.get_cost(a2, Y)

    assert np.isclose(expected_cost, cost, rtol=1e-6), f"Incorrect: Expected {expected_cost}; Got {cost}"

  def test_backward_propagation(self):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = (np.random.randn(1, 3) > 0)

    parameters = {
      'W1': np.array(
        [
          [-0.00416758, -0.00056267],
          [-0.02136196,  0.01640271],
          [-0.01793436, -0.00841747],
          [ 0.00502881, -0.01245288]
        ]
      ),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[0.],[0.],[0.],[0.]]),
     'b2': np.array([[0.]])}

    cache = {
      'A1': np.array(
        [
          [-0.00616578,  0.0020626 ,  0.00349619],
          [-0.05225116,  0.02725659, -0.02646251],
          [-0.02009721,  0.0036869 ,  0.02883756],
          [ 0.02152675, -0.01385234,  0.02599885]
        ]
      ),
      'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
      'Z1': np.array(
        [
          [-0.00616586,  0.0020626 ,  0.0034962 ],
          [-0.05229879,  0.02726335, -0.02646869],
          [-0.02009991,  0.00368692,  0.02884556],
          [ 0.02153007, -0.01385322,  0.02600471]
        ]
      ),
      'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])
    }

    expected_grads = {
      'dW1': np.array(
        [
          [ 0.00301023, -0.00747267],
          [ 0.00257968, -0.00641288],
          [-0.00156892,  0.003893  ],
          [-0.00652037,  0.01618243]
        ]
      ),
      'db1': np.array(
        [
          [ 0.00176201],
          [ 0.00150995],
          [-0.00091736],
          [-0.00381422]
        ]
      ),
      'dW2': np.array(
        [[0.00078841, 0.01765429, -0.00084166, -0.01022527]]
      ),
      'db2': np.array(
        [[-0.16655712]]
      )
    }

    model = PlanarClassifier()
    model.set_parameters(parameters)
    grads = model.backward_propagation(cache, X, Y)

    for key in expected_grads:
      np.testing.assert_allclose(
        grads[key],
        expected_grads[key],
        rtol=1e-8,
        atol=1e-8,
        err_msg=f"Incorrect gradient for {key}. Expected {expected_grads[key]}; Got {grads[key]}"
      )

  def test_update_parameters(self):
    parameters = {
      'W1': np.array(
        [
          [-0.00615039,  0.0169021 ],
          [-0.02311792,  0.03137121],
          [-0.0169217 , -0.01752545],
          [ 0.00935436, -0.05018221]
        ]
      ),
      'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
      'b1': np.array(
        [
          [ -8.97523455e-07],
          [  8.15562092e-06],
          [  6.04810633e-07],
          [ -2.54560700e-06]
        ]
      ),
      'b2': np.array([[  9.14954378e-05]])
    }

    grads = {
      'dW1': np.array(
        [
          [ 0.00023322, -0.00205423],
          [ 0.00082222, -0.00700776],
          [-0.00031831,  0.0028636 ],
          [-0.00092857,  0.00809933]
        ]
      ),
      'dW2': np.array([[-1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03]]),
      'db1': np.array(
        [
          [  1.05570087e-07],
          [ -3.81814487e-06],
          [ -1.90155145e-07],
          [  5.46467802e-07]
        ]
      ),
      'db2': np.array([[ -1.08923140e-05]])
    }

    model = PlanarClassifier()
    model.set_parameters(parameters)
    learning_rate = 1.2
    updated_parameters = model.update_parameters(grads, learning_rate)

    expected_parameters = {
      'W1': np.array(
        [
          [-0.00643025,  0.01936718],
          [-0.02410458,  0.03978052],
          [-0.01653973, -0.02096177],
          [ 0.01046864, -0.05990141]
        ]
      ),
      'b1': np.array(
        [
          [-1.02420756e-06],
          [ 1.27373948e-05],
          [ 8.32996807e-07],
          [-3.20136836e-06]
        ]
      ),
      'W2': np.array(
        [[-0.01041081, -0.04463285,  0.01758031,  0.04747113]]
      ),
      'b2': np.array([[0.00010457]])
    }

    for key in expected_parameters:
      np.testing.assert_allclose(
        updated_parameters[key],
        expected_parameters[key],
        rtol=1e-8,
        atol=1e-8,
        err_msg=f"Incorrect gradient for {key}. Expected {expected_parameters[key]}; Got {updated_parameters[key]}"
      )