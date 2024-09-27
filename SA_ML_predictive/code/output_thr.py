from utils import data_and_labels_from_indices, reshape_for_model, convert_list, TMP_MODELS_PATH
from automate_training import BATCH_SIZE, hyperparameter_numcells, hyperparameter_kernel_size
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
PREDS_PATH = "../predictions/"

def predict_for_validation(model_name, test_number, train_and_validation_data, train_and_validation_labels, kfold_second):
    params_nr = 0   
    
    if not os.path.isdir(TMP_MODELS_PATH + model_name):
      os.makedirs(TMP_MODELS_PATH + model_name)

    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
        indices.append([train_data_indices, validation_data_indices])
     
    for numcells in hyperparameter_numcells: 
      for kernel in hyperparameter_kernel_size:   
         params_nr += 1 
         for i, pair in enumerate(indices):   
            
            # Load correct model.
            model_used = tf.keras.models.load_model(TMP_MODELS_PATH + model_name + "/" + model_name + "_test_" + str(test_number) + "_fold_" + str(i + 1) + "_params_" + str(params_nr) + "_num_cells_" + str(numcells) + "_kernel_size_" + str(kernel) + ".h5")
          
            validation_data_indices = pair[1]

            # Convert validation indices to validation data and validation labels.
            val_data, val_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
    
            # Get model predictions on the validation data.
            val_data, val_labels = reshape_for_model(model_name, val_data, val_labels)
            model_predictions = model_used.predict(val_data, batch_size = BATCH_SIZE)
            model_predictions = convert_list(model_predictions)  

            if not os.path.isdir(PREDS_PATH + model_name):
               os.makedirs(PREDS_PATH + model_name)

            file_predictions = PREDS_PATH + model_name + "/predictions_" + model_name + "_test_" + str(test_number) + "_fold_" + str(i + 1) + "_params_" + str(params_nr) + "_num_cells_" + str(numcells) + "_kernel_size_" + str(kernel) + ".txt"
            open_file = open(file_predictions, "w") 
            open_file.write(str(model_predictions))
            open_file.close()
 
            file_labels = PREDS_PATH + model_name + "/labels_" + model_name + "_test_" + str(test_number) + "_fold_" + str(i + 1) + "_params_" + str(params_nr) + "_num_cells_" + str(numcells) + "_kernel_size_" + str(kernel) + ".txt"
            open_file = open(file_labels, "w") 
            open_file.write(str(list(val_labels)))
            open_file.close()

def weird_division(n, d):
    return n / d if d else 0
   
# Convert probability to class based on the threshold of probability.
def convert_to_binary(model_predictions, threshold = 0.5):
    model_predictions_binary = []

    for x in model_predictions:
        if x >= threshold:
            model_predictions_binary.append(1.0)
        else:
            model_predictions_binary.append(0.0)

    return model_predictions_binary

def return_GMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1
    
    return np.sqrt(tp / apo * tn / ane)

# Count correct predictions based on a custom threshold of probability.
def my_accuracy_calculate(test_labels, model_predictions, threshold = 0.5):
    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):

        if model_predictions[i] == test_labels[i]:
            score += 1

    return score / len(test_labels) * 100

def read_PR_ROC(model_name, numcells, kernel):
   test_labels = []
   model_predictions = []
   print("%s num_cells: %d kernel_size: %d" % (model_name, numcells, kernel))
   for model_predictions_file in os.listdir(PREDS_PATH + model_name):
      if "predictions_" in model_predictions_file and "_num_cells_" + str(numcells) + "_kernel_size_" + str(kernel) in model_predictions_file:
         open_file = open(PREDS_PATH + model_name + "/" + model_predictions_file, "r") 
         lines = open_file.readlines()
         open_file.close() 
         model_predictions_one_part = eval(lines[0].replace("\n", "").strip())
         for p in model_predictions_one_part:
            model_predictions.append(p)
         open_file = open(PREDS_PATH + model_name + "/" + model_predictions_file.replace("predictions", "labels"), "r") 
         lines = open_file.readlines()
         open_file.close() 
         model_labels_one_part = eval(lines[0].replace("\n", "").strip())
         for l in model_labels_one_part:
            test_labels.append(int(l))

   # Get recall and precision.
   precision, recall, thresholdsPR = precision_recall_curve(test_labels, model_predictions)

   # Get false positive rate and true positive rate.
   fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

   # Calculate the F1 score for each threshold.
   fscore = []
   for i in range(len(precision)):
      fscore.append(weird_division(2 * precision[i] * recall[i], precision[i] + recall[i]))

   # Locate the index of the largest F1 score.
   ixPR = np.argmax(fscore) 

   # Calculate the g-mean for each threshold.
   gmeans = np.sqrt(tpr * (1 - fpr))

   # Locate the index of the largest g-mean.
   ixROC = np.argmax(gmeans)

   model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholdsPR[ixPR])
   model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholdsROC[ixROC])
   model_predictions_binary = convert_to_binary(model_predictions, 0.5)
 
   print("PR thr", thresholdsPR[ixPR])
   print("PR AUC", auc(recall, precision)) 

   print("ROC thr", thresholdsROC[ixROC])
   print("ROC AUC", roc_auc_score(test_labels, model_predictions)) 

   print("F1 (0.5)", f1_score(test_labels, model_predictions_binary))
   print("F1 (PR thr)", f1_score(test_labels, model_predictions_binary_thrPR_new))
   print("F1 (ROC thr)", f1_score(test_labels, model_predictions_binary_thrROC_new))

   print("gmean (0.5)", return_GMEAN(test_labels, model_predictions_binary))
   print("gmean (PR thr)", return_GMEAN(test_labels, model_predictions_binary_thrPR_new))
   print("gmean (ROC thr)", return_GMEAN(test_labels, model_predictions_binary_thrROC_new))

   print("Accuracy (0.5)", my_accuracy_calculate(test_labels, model_predictions, 0.5))
   print("Accuracy (PR thr)", my_accuracy_calculate(test_labels, model_predictions, thresholdsPR[ixPR]))
   print("Accuracy (ROC thr)", my_accuracy_calculate(test_labels, model_predictions, thresholdsROC[ixROC]))