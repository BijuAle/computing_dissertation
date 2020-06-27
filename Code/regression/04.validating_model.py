import numpy as np
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt, rc

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

def get_features_targets(data):
  features = np.zeros((data.shape[0], 4)) #n lines, 4 columns
  features[:,0] = data['u'] - data['g']
  features[:,1] = data['g'] - data['r']
  features[:,2] = data['r'] - data['i']
  features[:,3] = data['i'] - data['z']
  targets = data['redshift']
  return (features, targets)

def median_diff(predicted, actual):
  diff = np.median(np.absolute(predicted - actual))
  plot_med_hst(predicted, actual)
  return diff

# Show a histogram of median distribution
def plot_med_hst(predicted, actual):  
  d=[]
  for i in range(len(predicted)):
    d.append(abs(predicted[i]-actual[i]))
  plt.hist(d, bins='auto')
  plt.xlabel('Median Residual')
  plt.ylabel('Frequency')
  plt.xlim(0, .5)
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  plt.show()

# performing hold-out validation
# training the model and returning the
#  prediction accuracy with median_diff
def validate_model(model, features, targets):
  # splitting the data into
  #  training and testing features and predictions
  split = features.shape[0]//2
  train_features = features[:split]
  test_features = features[split:]
  
  train_targets = targets[:split]
  test_targets = targets[split:]
  
  # training the model
  model.fit(train_features, train_targets)

  # get the predicted_redshifts
  predictions = model.predict(test_features)
  
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions), predictions, test_targets


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  # initialize model
  dtr = DecisionTreeRegressor()

  # validate the model and print the med_diff
  diff, predictions, targets = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))

  print(predictions)
  print(targets)
  print(predictions.shape)
  print(targets.shape)
  plt.scatter(targets, predictions, s=0.4)
  plt.xlim((0, targets.max()))
  plt.ylim((0, predictions.max()))
  plt.xlabel('Measured Redshift')
  plt.ylabel('Predicted Recleardshift')
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  #plt.show()