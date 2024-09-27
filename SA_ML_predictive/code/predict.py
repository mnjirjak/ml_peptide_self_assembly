import numpy as np
import sys  
from utils import load_data, merge_data, reshape_for_model, convert_list, MAX_LEN, MODELS_PATH, ALL_MODELS
from automate_training import BATCH_SIZE
import tensorflow as tf

# Call this script from the command line to generate predictions for one peptide.
# The first argument, listed after --sequence, is the peptide sequence (no longer than the max length, currently 24).
# The second argument, listed after --ml-model, is the model name, one of "AP", "SP", "TSNE_SP", "TSNE_AP_SP" or "AP_SP".
# If no model is specified, the "AP_SP" model is used.
 
if len(sys.argv) > 2 and len(sys.argv[2]) <= MAX_LEN:
    pep_list = [sys.argv[2]] 
    model = "AP_SP"
    if len(sys.argv) > 4 and sys.argv[4] in ALL_MODELS:
        model = sys.argv[4] 
    else:
        if len(sys.argv) <= 4:
            print("No model selected, using", model, "model")
        if len(sys.argv) > 4 and sys.argv[4] not in ALL_MODELS:
            print("Model", sys.argv[4], "does not exist, using", model, "model")
        
    seq_example = ""
    for i in range(MAX_LEN):
        seq_example += "A"
    pep_list.append(seq_example)
    pep_labels = ["1", "1"]
 
    best_model = "" 

    offset = 1

    properties = np.ones(95)
    properties[0] = 0
    mask_value = 2
  
    SA, NSA = load_data(model, [pep_list, pep_labels], offset, properties, mask_value)
    all_data, all_labels = merge_data(SA, NSA) 
    
    # Load the best model.
    best_model = tf.keras.models.load_model(MODELS_PATH + model + ".h5")
 
    # Get model predictions on the test data.
    test_data, test_labels = reshape_for_model(model, all_data, all_labels)
    model_predictions = best_model.predict(test_data, batch_size = BATCH_SIZE)
    model_predictions = convert_list(model_predictions)  

    print(model_predictions[0])
else:
    if len(sys.argv) <= 2:
        print("No peptide")
    if len(sys.argv[2]) > MAX_LEN:
        print("Peptide", sys.argv[2], "is too long, the maximum peptide lenght for the model is", MAX_LEN)