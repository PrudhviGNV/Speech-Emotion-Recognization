from utilities import load_data
from utilities import extract features
from utilities import int2emotion,AVAILABLEEMOTIONS


X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
