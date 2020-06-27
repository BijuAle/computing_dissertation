import numpy as np
from sklearn.tree import DecisionTreeRegressor

def get_features_targets(data):
  #n lines, 4 columns
  features = np.zeros((data.shape[0], 4)) 
  features[:,0] = data['u'] - data['g']
  features[:,1] = data['g'] - data['r']
  features[:,2] = data['r'] - data['i']
  features[:,3] = data['i'] - data['z']
  targets = data['redshift']
  return (features, targets)

# loading the data and
data = np.load('sdss_galaxy_colors.npy')
# generating the features and targets
features, targets = get_features_targets(data)
  
# initializing model
dtr = DecisionTreeRegressor()

# training the model
dtr.fit(features, targets)

# making predictions using the same features
predictions = dtr.predict(features)

# printing the first 4 predicted redshifts
print(predictions[:4])

