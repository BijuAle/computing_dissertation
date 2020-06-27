import numpy as np
import math

def splitdata_train_test(data, fraction_training):
  
  # randomizing data set order
  np.random.seed(0)
  np.random.shuffle(data)

  # find split point
  training_rows = math.floor(len(data) * fraction_training)
  
  training_set = data[:training_rows]
  testing_set = data[training_rows:len(data)]
  return (training_set, testing_set)

if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # setting the fraction of data which should be in the training set
  fraction_training = 0.7

  # spliting the data
  training, testing = splitdata_train_test(data, fraction_training)

  # printing the key values
  print('Number of galaxies in dataset:', len(data))
  print('Train fraction:', fraction_training)
  print('Number of galaxies in training set:', len(training))
  print('Number of galaxies in testing set:', len(testing))

  # print an example
  print('Field and value for 1 Galaxy:')
  for name, value in zip(data.dtype.names, data[0]):
    print('{:20} {:.6}'.format(name, value))