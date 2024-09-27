import numpy as np
from automate_training import hyperparameter_tuning     
from utils import get_seed, load_data, merge_data, data_and_labels_from_indices, ALL_MODELS
from sklearn.model_selection import StratifiedKFold
import sys

# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
EPOCHS = 70
offset = 1
# Define random seed
seed = get_seed()
SA_data = 'data_SA.csv'
properties = np.ones(95)
properties[0] = 0
mask_value = 2
model_name = "AP_SP" 

if len(sys.argv) < 3:
    print("No model selected, using", model_name, "model")
if len(sys.argv) >= 3 and sys.argv[2] not in ALL_MODELS:
    print("Model", sys.argv[2], "does not exist, using", model_name, "model")
if len(sys.argv) >= 3 and sys.argv[2] in ALL_MODELS:
    model_name = sys.argv[2]

SA, NSA = load_data(model_name, SA_data, offset, properties, mask_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not.  
# During model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data(SA, NSA)

# Define N-fold cross validation test harness for splitting the test data from the train and validation data.
kfold_first = StratifiedKFold(n_splits = N_FOLDS_FIRST, shuffle = True, random_state = seed)
# Define N-fold cross validation test harness for splitting the validation from the train data.
kfold_second = StratifiedKFold(n_splits = N_FOLDS_SECOND, shuffle = True, random_state = seed) 
  
test_number = 0

for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
    test_number += 1
       
    # Convert train and validation indices to train and validation data and train and validation labels.
    train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices) 
    
    # Convert test indices to test data and test labels.
    test_data, test_labels = data_and_labels_from_indices(all_data, all_labels, test_data_indices)
  
    # Train the model.
    hyperparameter_tuning(model_name, test_number, train_and_validation_data, train_and_validation_labels, kfold_second, EPOCHS, factor_NSA, mask_value = mask_value)