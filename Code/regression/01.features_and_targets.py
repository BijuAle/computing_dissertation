import numpy as np

def get_features_targets(data):
  #n rows, 4 columns
  features = np.zeros((data.shape[0], 4)) 
  features[:,0] = data['u'] - data['g']
  features[:,1] = data['g'] - data['r']
  features[:,2] = data['r'] - data['i']
  features[:,3] = data['i'] - data['z']
   
  targets = data['redshift']
  
  return (features, targets)

if __name__ == "__main__":
  # loading the data
  data = np.load('sdss_galaxy_colors.npy')

  features, targets = get_features_targets(data)
    
  # printing the shape of the returned arrays
  print(features[:2])
  print(targets[:2])