import numpy as np

from NeuralCatClassifier.utilities import sigmoid

class NeuralCatIdentifier:
  def __init__(self, dim):
    self.dim = dim

    # already performing initialisation
    self.w = np.zeros((self.dim, 1), dtype=float)
    self.b = 0.0
    self.dw = np.zeros_like(self.w)
    self.db = 0.0 
  

  def get_initialised_params(self):
    """
      This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
      
      Argument:
      dim -- size of the w vector we want (or number of parameters in this case)
      
      Returns:
      w -- initialized vector of shape (dim, 1)
      b -- initialized scalar (corresponds to the bias)
    """
    return self.w, self.b
  

  def propagate(self, X, Y):
    """
      Implement the cost function and its gradient for the propagation explained above

      Arguments:
      w -- weights, a numpy array of size (num_px * num_px * 3, 1)
      b -- bias, a scalar
      X -- data of size (num_px * num_px * 3, number of examples)
      Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

      Return:
      cost -- negative log-likelihood cost for logistic regression
      dw -- gradient of the loss with respect to w, thus same shape as w
      db -- gradient of the loss with respect to b, thus same shape as b
      
      Tips:
      - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # forward passs
    z = np.dot(self.w.T, X) + self.b
    A = sigmoid(z)
    cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.squeeze(cost)
    
    # backward pass - backward propagation
    dw = (1/m) * (np.dot(X, (A - Y).T))
    db = (1/m) * (np.sum(A - Y))
    
    grads = {
      "dw": dw,
      "db": db
    }
    
    return grads, cost
  

  def optimize(self, X, Y, num_iterations, learning_rate, print_cost = False):
    """
      This function optimizes w and b by running a gradient descent algorithm
      
      Arguments:
      w -- weights, a numpy array of size (num_px * num_px * 3, 1)
      b -- bias, a scalar
      X -- data of shape (num_px * num_px * 3, number of examples)
      Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
      num_iterations -- number of iterations of the optimization loop
      learning_rate -- learning rate of the gradient descent update rule
      print_cost -- True to print the loss every 100 steps
      
      Returns:
      params -- dictionary containing the weights w and bias b
      grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
      costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
      
      Tips:
      You basically need to write down two steps and iterate through them:
          1) Calculate the cost and the gradient for the current parameters. Use propagate().
          2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
      grads, cost = self.propagate(X, Y)
      
      self.dw = grads["dw"]
      self.db = grads["db"]
      
      self.w -= (learning_rate * self.dw)
      self.b -= (learning_rate * self.db)
      
      if i % 100 == 0:
        costs.append(cost)
      
      if print_cost and i % 100 == 0:
        print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {
      "w": self.w,
      "b": self.b
    }
    
    grads = {
      "dw": self.dw,
      "db": self.db
    }
    
    return params, grads, costs
  

  def predict(self, X):
    '''
      Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
      
      Arguments:
      w -- weights, a numpy array of size (num_px * num_px * 3, 1)
      b -- bias, a scalar
      X -- data of size (num_px * num_px * 3, number of examples)
      
      Returns:
      Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = self.w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + self.b)
    Y_prediction = (A >= 0.5) * 1.0
    # assert(Y_prediction.shape == (1, m))
    return Y_prediction
  

  def run_model(self, X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
      Builds the logistic regression model by calling the function you've implemented previously
      
      Arguments:
      X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
      Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
      X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
      Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
      num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
      learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
      print_cost -- Set to true to print the cost every 100 iterations
      
      Returns:
      d -- dictionary containing information about the model.
    """
    
    
    # initialize parameters with zeros
    w, b = self.get_initialised_params()
    
    parameters, _, costs = self.optimize(
      X_train,
      Y_train,
      num_iterations,
      learning_rate,
      print_cost
    )
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = self.predict(X_test)
    Y_prediction_train = self.predict(X_train)

    # Print train/test Errors
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

    # results
    model_outputs = {
      "costs": costs,
      "Y_prediction_test": Y_prediction_test, 
      "Y_prediction_train" : Y_prediction_train, 
      "w" : w, 
      "b" : b,
      "learning_rate" : learning_rate,
      "num_iterations": num_iterations,
      "test_accuracy": test_accuracy,
      "train_accuracy": train_accuracy
    }
    
    return model_outputs