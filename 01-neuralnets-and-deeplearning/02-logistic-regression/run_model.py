from NeuralCatClassifier import NeuralCatIdentifier, DataPreprocessor
from NeuralCatClassifier.utilities import (
  load_dataset
)


def main():

  preprocessor = DataPreprocessor()

  dataset = load_dataset()
  if dataset is None:
    print("Dataset loading failed. Exiting...")
  else:
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, _ = dataset
    print("Dataset loaded successfully!\n")

  train_set_x_flatten = preprocessor.flatten_data(train_set_x_orig)
  test_set_x_flatten = preprocessor.flatten_data(test_set_x_orig)

  train_set_x = train_set_x_flatten/255.
  test_set_x = test_set_x_flatten/255.


  cat_classifier = NeuralCatIdentifier(train_set_x.shape[0])
  model_outputs = cat_classifier.run(
    train_set_x,
    train_set_y, test_set_x,
    test_set_y,
    num_iterations = 2000,
    learning_rate = 0.005,
    print_cost = False
  )

  print(
    f"Model Outputs:\n"
    f"train accuracy: {model_outputs['train_accuracy']}\n"
    f"test accuracy: {model_outputs['test_accuracy']}\n"
  )


if __name__ == "__main__":
  main()

