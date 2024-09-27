from seqprops import SequentialPropertiesEncoder  
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import pandas as pd

DATA_PATH = './SA_ML_predictive/data/'
TMP_MODELS_PATH = './SA_ML_predictive/models/tmp_models/'
MODELS_PATH = './SA_ML_predictive/models/'
MAX_LEN = 24 
ALL_MODELS = ["AP", "SP", "TSNE_SP", "TSNE_AP_SP", "AP_SP"]

def set_seed(x):
    seed_file = open(DATA_PATH + "seed.txt", "w")
    seed_file.write(str(x))
    seed_file.close()
 
def get_seed():
    seed_file = open(DATA_PATH + "seed.txt", "r")
    line = seed_file.readlines()[0].replace("\n", "")
    seed_file.close()
    return int(line)

def randomize_seed():
    set_seed(np.random.randint(1000000000))

def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i]) 
        
    return data, labels 

def scale(AP_dictionary, offset = 1):
    data = [AP_dictionary[key] for key in AP_dictionary]

    # Determine min and max AP scores.
    min_val = min(data)
    max_val = max(data)

    # Scale AP scores to range [- offset, offset].
    for key in AP_dictionary:
        AP_dictionary[key] = (AP_dictionary[key] - min_val) / (max_val - min_val) * 2 * offset - offset
   
def padding(array, len_to_pad, value_to_pad):

    # Fill array with padding value to maximum length.
    new_array = [value_to_pad for i in range(len_to_pad)]
    for val_index in range(len(array)):

        # Add original values.
        if val_index < len(new_array):
            new_array[val_index] = array[val_index]
    return new_array

def split_amino_acids(sequence, amino_acids_AP_scores):
    ap_list = []

    # Replace each amino acid in the sequence with a corresponding AP score.
    for letter in sequence:
        ap_list.append(amino_acids_AP_scores[letter])

    return ap_list

def split_dipeptides(sequence, dipeptides_AP_scores):
    ap_list = []

    # Replace each dipeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 1):
        ap_list.append(dipeptides_AP_scores[sequence[i:i + 2]])

    return ap_list

def split_tripeptides(sequence, tripeptides_AP_scores):
    ap_list = []

    # Replace each tripeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 2):
        ap_list.append(tripeptides_AP_scores[sequence[i:i + 3]])

    return ap_list

def reshape_for_model(model_name, all_data, all_labels):

    labels = []
    for i in range(len(all_data)):
        labels.append(all_labels[i]) 
    if len(labels) > 0:
        labels = np.array(labels)

    if "SP" in model_name and "AP" not in model_name:

        new_data = []
        for i in range(len(all_data)):
            new_data.append(np.array(all_data[i]))
        if len(new_data) > 0:
            new_data = np.array(new_data)
        return new_data, labels
     
    data = [[] for i in range(len(all_data[0]))]
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])

    new_data = []
    last_data = []    
    for i in range(len(data)):
        if len(data[i]) > 0 and i < 3:  
            new_data.append(np.array(data[i]))
        if "AP" in model_name and "SP" in model_name:
            if len(data[i]) > 0 and i >= 3: 
                last_data.append(np.array(data[i])) 

    if len(last_data) > 0:
        last_data = np.array(last_data).transpose(1, 2, 0)
        new_data.append(last_data)
        
    return new_data, labels 
   
def read_SA_data(SA_data):
    sequences = []
    labels = []

    csv_data = pd.read_csv(DATA_PATH + SA_data, sep = ";", index_col = False) 
    for i, peptide in enumerate(csv_data["peptide_sequence"]): 
        if len(peptide) <= MAX_LEN and (str(csv_data["peptide_label"][i]) == "0" or str(csv_data["peptide_label"][i]) == "1"): 
            sequences.append(peptide)
            labels.append(str(csv_data["peptide_label"][i])) 
    return sequences, labels

def load_data_AP(offset = 1):
    # Load AP scores. 
    amino_acids_AP = np.load(DATA_PATH + 'amino_acids_AP.npy', allow_pickle = True).item()
    dipeptides_AP = np.load(DATA_PATH + 'dipeptides_AP.npy', allow_pickle = True).item()
    tripeptides_AP = np.load(DATA_PATH + 'tripeptides_AP.npy', allow_pickle = True).item()
    
    # Scale scores to range [-offset, offset].
    scale(amino_acids_AP, offset)
    scale(dipeptides_AP, offset)
    scale(tripeptides_AP, offset)

    return amino_acids_AP, dipeptides_AP, tripeptides_AP
 
def load_data(model_name, SA_data, offset = 1, properties_to_include = [], mask_value = 2):
    if ".csv" in SA_data:
        sequences, labels = read_SA_data(SA_data)
    else:
        sequences, labels = SA_data[0], SA_data[1]
            
    if "SP" in model_name and "TSNE" not in model_name:
        # Encode sequences
        encoder = SequentialPropertiesEncoder(scaler = MinMaxScaler(feature_range = (-offset, offset)))
        encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []

        if "AP" in model_name:
        
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAX_LEN + 1 * ("SP" in model_name and "TSNE" not in model_name), mask_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAX_LEN + 1 * ("SP" in model_name and "TSNE" not in model_name), mask_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAX_LEN + 1 * ("SP" in model_name and "TSNE" not in model_name), mask_value)  

            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 

        if "SP" in model_name and "TSNE" not in model_name:
             
            other_props = np.transpose(encoded_sequences[index])  

            for prop_index in range(len(properties_to_include)):
                if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                    array = other_props[prop_index]
                    for i in range(len(array)):
                        if i >= len(sequences[index]):
                            array[i] = mask_value
                    new_props.append(array) 
            
            if "AP" not in model_name:
                     
                new_props = np.transpose(new_props)
            
        if "SP" in model_name and "TSNE" in model_name:

            feature_dict_1 = np.load(DATA_PATH + 'TSNE_SP_1_component.npy', allow_pickle = True).item()
            feature_dict_2 = np.load(DATA_PATH + 'TSNE_SP_2_components.npy', allow_pickle = True).item()
            feature_dict_3 = np.load(DATA_PATH + 'TSNE_SP_3_components.npy', allow_pickle = True).item()

            feature1 = split_amino_acids(sequences[index], feature_dict_1)
            feature1_padded = padding(feature1, MAX_LEN, mask_value)

            feature2 = split_amino_acids(sequences[index], feature_dict_2)
            feature2_padded = padding(feature2, MAX_LEN, mask_value)

            feature3 = split_amino_acids(sequences[index], feature_dict_3)
            feature3_padded = padding(feature3, MAX_LEN, mask_value)

            new_props.append(feature1_padded)
            new_props.append(feature2_padded)
            new_props.append(feature3_padded) 

        if str(labels[index]) == '1':
            SA.append(new_props) 
        elif str(labels[index]) == '0':
            NSA.append(new_props) 

    if len(SA) > 0:
        SA = np.array(SA)
    if len(NSA) > 0:
        NSA = np.array(NSA)

    return SA, NSA
 
def merge_data(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    if len(merged_data) > 0:
        merged_data = np.array(merged_data)
    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0
    return merged_data, merged_labels

def convert_list(model_predictions):
    new_predictions = []
    for j in range(len(model_predictions)):
        new_predictions.append(model_predictions[j][0])
    return new_predictions