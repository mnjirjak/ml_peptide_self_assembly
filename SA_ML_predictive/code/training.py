import numpy as np
from automate_training import model_training     
from utils import load_data, merge_data, ALL_MODELS
import sys

# Algorithm settings  
hyp_default_values = dict()
hyp_default_values["--num-cells"] = {"AP": 32, "SP": 64, "AP_SP": 32, "TSNE_SP": 48, "TSNE_AP_SP": 64} 
hyp_default_values["--kernel-size"] = {"AP": 4, "SP": 4, "AP_SP": 4, "TSNE_SP": 6, "TSNE_AP_SP": 6} 
EPOCHS = 70
offset = 1

SA_data = 'data_SA.csv'
properties = np.ones(95)
properties[0] = 0
mask_value = 2
model_name = "AP_SP"
 
found_a_model = False

if len(sys.argv) >= 2:

    for i in range(1, len(sys.argv)):

        parameter_name = sys.argv[i]

        if parameter_name == "--ml-model":

            if i + 1 < len(sys.argv):

                proposed_model_name = sys.argv[i + 1]

                if proposed_model_name not in ALL_MODELS:
                    print("Model", proposed_model_name, "does not exist, using the", model_name, "model")

                if proposed_model_name in ALL_MODELS:
                    model_name = proposed_model_name
                    found_a_model = True
                    print("Using the", model_name, "model")
                
if not found_a_model:
    print("No model selected, using the", model_name, "model")
    
hyperparameter_name_values = {"--num-cells": [32, 48, 64], "--kernel-size": [4, 6, 8]}

found_hyp = set()

if len(sys.argv) >= 2:

    for i in range(1, len(sys.argv)):
        
        hyperparameter_name = sys.argv[i]
 
        if hyperparameter_name in hyperparameter_name_values and hyperparameter_name not in found_hyp:

            if i + 1 < len(sys.argv):

                hyperparameter_value = sys.argv[i + 1]

                digits_all = True
                for c in hyperparameter_value:
                    if not c.isdigit(): 
                        digits_all = False
                        break

                if digits_all:
                    hyp_default_values[hyperparameter_name][model_name] = int(hyperparameter_value) 
                    found_hyp.add(hyperparameter_name)
                    print("Using a value of", hyp_default_values[hyperparameter_name][model_name], "for the hyperparameter", hyperparameter_name, "for the", model_name, "model")
                    if hyp_default_values[hyperparameter_name][model_name] not in hyperparameter_name_values[hyperparameter_name]:
                        str_suggested = ""
                        for val in hyperparameter_name_values[hyperparameter_name][:-1]:
                            str_suggested += str(val) + ", "
                        str_suggested += "and " + str(hyperparameter_name_values[hyperparameter_name][-1])
                        print("WARNING: A value of", hyp_default_values[hyperparameter_name][model_name], "for the hyperparameter", hyperparameter_name, "for the", model_name, "model is not among the suggested values, the suggested values are", str_suggested)
                    if model_name == "AP" and hyperparameter_name == "--kernel-size":
                        print("WARNING: The hyperparameter", hyperparameter_name, "is not used for the", model_name, "model and will be ignored.")
                else:
                    print("The value", hyperparameter_value, "is not an integer and is not supported for the hyperparameter", hyperparameter_name)
          
        else:
            if hyperparameter_name in found_hyp:
                print("WARNING: Duplicate definition for the hyperparameter", hyperparameter_name + ", using the first value of", hyp_default_values[hyperparameter_name][model_name])

            digits_all = True
            for c in hyperparameter_name:
                if not c.isdigit(): 
                    digits_all = False
                    break

            if hyperparameter_name not in hyperparameter_name_values and hyperparameter_name not in ALL_MODELS and not digits_all and not hyperparameter_name == "--ml-model":
                print("Unknown hyperparameter", hyperparameter_name)

for hyp_name in hyp_default_values:
    if hyp_name not in found_hyp and not (model_name == "AP" and hyp_name == "--kernel-size"):
        print("No value specified for the hyperparameter", hyp_name + ", using a default value of", hyp_default_values[hyp_name][model_name], "for the", model_name , "model")
  
SA, NSA = load_data(model_name, SA_data, offset, properties, mask_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not.  
# During model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)

# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data(SA, NSA)
    
# Train the model.
model_training(model_name, all_data, all_labels, hyp_default_values["--num-cells"][model_name], hyp_default_values["--kernel-size"][model_name], EPOCHS, factor_NSA, mask_value = mask_value)
