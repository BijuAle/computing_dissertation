import numpy as np

# calculating the median of the differences
# between our predicted and actual values
def median_diff(predicted, actual):
  diff = np.median(np.absolute(predicted - actual))
  return diff


if __name__ == "__main__":
  # load testing data
  targets = np.load('targets.npy')
  predictions = np.load('predictions.npy')

  # calling the function to measure
  #  the accuracy of the predictions
  diff = median_diff(predictions, targets)

  # print the median difference
  print("Median difference: {:0.3f}".format(diff))

